#include "jetson_pipeline.h"
#include <iostream>
#include <chrono>

namespace bcs {

JetsonBCSPipeline::JetsonBCSPipeline(const PipelineConfig& config)
    : config_(config) {}

JetsonBCSPipeline::~JetsonBCSPipeline() {
    stop();
}

bool JetsonBCSPipeline::initialize() {
    std::cout << "[JetsonBCSPipeline] Initializing components for Jetson Orin architecture..." << std::endl;

    decoder_ = std::make_unique<NVMMDecoder>(config_.video_source);
    if (!decoder_->initialize()) return false;

    yolo_engine_ = std::make_unique<TensorRTEngine>(config_.yolo_engine_path, false);
    if (!yolo_engine_->load_engine()) return false;

    dinov2_engine_ = std::make_unique<TensorRTEngine>(config_.dinov2_engine_path, true);
    if (!dinov2_engine_->load_engine()) return false;

    classifier_ = std::make_unique<BCSClassifier>(384, 128, 3);
    if (!classifier_->load_weights(config_.head_weights_path)) return false;

    std::cout << "[JetsonBCSPipeline] All engines loaded and zero-copy buffers bound." << std::endl;
    return true;
}

void JetsonBCSPipeline::start() {
    running_ = true;
    pipeline_thread_ = std::thread(&JetsonBCSPipeline::run_pipeline_loop, this);

    if (config_.enable_thermal_throttling) {
        thermal_thread_ = std::thread(&JetsonBCSPipeline::run_thermal_monitor, this);
    }
    if (config_.enable_watchdog) {
        watchdog_thread_ = std::thread(&JetsonBCSPipeline::run_watchdog, this);
    }
}

void JetsonBCSPipeline::stop() {
    if (running_) {
        running_ = false;
        if (pipeline_thread_.joinable()) pipeline_thread_.join();
        if (thermal_thread_.joinable()) thermal_thread_.join();
        if (watchdog_thread_.joinable()) watchdog_thread_.join();
        std::cout << "[JetsonBCSPipeline] Pipeline shut down gracefully." << std::endl;
    }
}

void JetsonBCSPipeline::run_pipeline_loop() {
    std::cout << "[JetsonBCSPipeline] Entering main inference loop..." << std::endl;
    int processed_count = 0;

    while (running_ && processed_count < 1000) {
        auto start_t = std::chrono::high_resolution_clock::now();

        FrameMetadata frame;
        if (!decoder_->read_frame(frame)) break;

        last_frame_timestamp_ms_ = frame.timestamp_ms;

        // 1. Detect cow via YOLOv8 TensorRT
        std::vector<DetectionBox> detections;
        yolo_engine_->execute_yolo(frame.nvmm_buffer_ptr, detections);

        // 2. For each detected cow, run DINOv2 backbone & BCS Classifier
        for (const auto& det : detections) {
            std::vector<float> mock_crop(3 * 224 * 224, 0.5f);
            std::vector<float> embedding;

            dinov2_engine_->execute_dinov2(mock_crop, embedding);
            ClassificationResult result = classifier_->predict(embedding);

            if (processed_count % 100 == 0) {
                std::cout << "[Frame " << frame.frame_id << "] Cow detected (Conf: "
                          << det.confidence << ") -> BCS Score: " << result.label
                          << " (Prob: " << result.confidence << ")" << std::endl;
            }
        }

        processed_count++;

        // Rate limiting according to FPS limit
        auto end_t = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();
        double target_ms = 1000.0 / current_fps_limit_.load();

        if (elapsed_ms < target_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(target_ms - elapsed_ms)));
        }
    }

    running_ = false;
}

void JetsonBCSPipeline::run_thermal_monitor() {
    std::cout << "[Daemon] Thermal Throttling Monitor started." << std::endl;
    while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        // Pseudo thermal reading for demo
        int current_temp = 58; 
        if (current_temp > 75 && current_fps_limit_ > 15) {
            std::cout << "[WARNING] Temperature exceeded 75C. Lowering target FPS to 15." << std::endl;
            current_fps_limit_ = 15;
        } else if (current_temp < 65 && current_fps_limit_ < 30) {
            current_fps_limit_ = 30;
        }
    }
}

void JetsonBCSPipeline::run_watchdog() {
    std::cout << "[Daemon] Decoder Watchdog started." << std::endl;
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        // Check hardware stream health
    }
}

} // namespace bcs
