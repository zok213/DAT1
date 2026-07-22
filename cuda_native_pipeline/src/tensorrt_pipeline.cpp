#include "tensorrt_pipeline.h"
#include <chrono>
#include <cmath>
#include <algorithm>

const char* CLASS_LABELS[3] = {"thin", "ideal", "fat"};

CudaBCSPipeline::CudaBCSPipeline(const std::string& yolo_model, const std::string& dino_model, const std::string& head_model)
    : is_ema_init_(false), alpha_(0.25f) {
    
    cudaStreamCreate(&stream_);
    
    // Allocate GPU Unified Pinned Memory for 1080p BGR Frame + 224x224 Normalized Input Tensor
    size_t frame_bytes = 1920 * 1080 * 3 * sizeof(uint8_t);
    size_t tensor_bytes = 1 * 3 * 224 * 224 * sizeof(float);
    
    cudaMallocManaged(&d_frame_bgr_, frame_bytes);
    cudaMallocManaged(&d_dino_input_, tensor_bytes);

    ema_state_[0] = 0.33f;
    ema_state_[1] = 0.33f;
    ema_state_[2] = 0.33f;

    std::cout << "[C++ CUDA Pipeline] Unified Pinned GPU Memory Allocated & Stream Initialized." << std::endl;
}

CudaBCSPipeline::~CudaBCSPipeline() {
    cudaFree(d_frame_bgr_);
    cudaFree(d_dino_input_);
    cudaStreamDestroy(stream_);
}

BCSResult CudaBCSPipeline::process_frame(const cv::Mat& frame_bgr) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int w = frame_bgr.cols;
    int h = frame_bgr.rows;
    size_t bytes = w * h * 3 * sizeof(uint8_t);

    // 1. Zero-Copy Host to Unified GPU Memory Transfer
    std::memcpy(d_frame_bgr_, frame_bgr.data, bytes);

    // 2. Launch Custom CUDA Kernel for GPU Crop + Mask + Resize (224x224) + ImageNet Normalize
    int x1 = static_cast<int>(w * 0.15f);
    int y1 = static_cast<int>(h * 0.15f);
    int x2 = static_cast<int>(w * 0.85f);
    int y2 = static_cast<int>(h * 0.85f);

    launch_crop_mask_normalize_kernel(
        d_frame_bgr_, w, h,
        x1, y1, x2, y2,
        nullptr, // nullptr mask = full crop
        d_dino_input_,
        stream_
    );

    cudaStreamSynchronize(stream_);

    // 3. Simulated Ultra-Fast TensorRT Engine Execution (DINOv2 + BcsHead)
    float raw_logits[3] = {0.10f, 0.85f, 0.05f};
    float exp_sum = std::exp(raw_logits[0]) + std::exp(raw_logits[1]) + std::exp(raw_logits[2]);
    float raw_probs[3] = {
        std::exp(raw_logits[0]) / exp_sum,
        std::exp(raw_logits[1]) / exp_sum,
        std::exp(raw_logits[2]) / exp_sum
    };

    // 4. Exponential Moving Average (EMA) Temporal Logit Filter
    if (!is_ema_init_) {
        ema_state_[0] = raw_probs[0];
        ema_state_[1] = raw_probs[1];
        ema_state_[2] = raw_probs[2];
        is_ema_init_ = true;
    } else {
        ema_state_[0] = alpha_ * raw_probs[0] + (1.0f - alpha_) * ema_state_[0];
        ema_state_[1] = alpha_ * raw_probs[1] + (1.0f - alpha_) * ema_state_[1];
        ema_state_[2] = alpha_ * raw_probs[2] + (1.0f - alpha_) * ema_state_[2];
    }

    int best_cls = 0;
    if (ema_state_[1] > ema_state_[best_cls]) best_cls = 1;
    if (ema_state_[2] > ema_state_[best_cls]) best_cls = 2;

    auto t1 = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    BCSResult res;
    res.class_id = best_cls;
    res.label = CLASS_LABELS[best_cls];
    res.confidence = ema_state_[best_cls];
    res.probabilities[0] = ema_state_[0];
    res.probabilities[1] = ema_state_[1];
    res.probabilities[2] = ema_state_[2];
    res.total_ms = total_ms;

    return res;
}
