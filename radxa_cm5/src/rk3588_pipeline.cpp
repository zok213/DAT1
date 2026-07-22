#include "rk3588_pipeline.h"
#include <iostream>
#include <chrono>

namespace rk3588 {

RK3588BCSPipeline::RK3588BCSPipeline(const PipelineConfig& config)
    : config_(config) {}

RK3588BCSPipeline::~RK3588BCSPipeline() {
    stop();
}

bool RK3588BCSPipeline::initialize() {
    std::cout << "[RK3588BCSPipeline] Initializing components for Radxa CM5 (RK3588)..." << std::endl;

    decoder_ = std::make_unique<MPPDecoder>(config_.video_source);
    if (!decoder_->initialize()) return false;

    rga_resizer_ = std::make_unique<RGAResizer>();
    if (!rga_resizer_->initialize()) return false;

    yolo_npu_ = std::make_unique<RKNNEngine>(config_.yolo_rknn_path);
    if (!yolo_npu_->load_model()) return false;

    dinov2_npu_ = std::make_unique<RKNNEngine>(config_.dinov2_rknn_path);
    if (!dinov2_npu_->load_model()) return false;

    classifier_ = std::make_unique<BCSClassifier>(384, 128, 3);
    if (!classifier_->load_weights(config_.head_weights_path)) return false;

    std::cout << "[RK3588BCSPipeline] MPP, RGA, and 3-Core RKNN NPU pipeline ready." << std::endl;
    return true;
}

void RK3588BCSPipeline::start() {
    running_ = true;

    decode_thread_ = std::thread(&RK3588BCSPipeline::decode_stage, this);
    rga_thread_ = std::thread(&RK3588BCSPipeline::rga_stage, this);
    npu_thread_ = std::thread(&RK3588BCSPipeline::npu_stage, this);

    if (config_.enable_thermal_throttling) {
        thermal_thread_ = std::thread(&RK3588BCSPipeline::thermal_monitor, this);
    }
    if (config_.enable_watchdog) {
        watchdog_thread_ = std::thread(&RK3588BCSPipeline::watchdog_daemon, this);
    }
}

void RK3588BCSPipeline::stop() {
    if (running_) {
        running_ = false;
        cv_.notify_all();

        if (decode_thread_.joinable()) decode_thread_.join();
        if (rga_thread_.joinable()) rga_thread_.join();
        if (npu_thread_.joinable()) npu_thread_.join();
        if (thermal_thread_.joinable()) thermal_thread_.join();
        if (watchdog_thread_.joinable()) watchdog_thread_.join();

        std::cout << "[RK3588BCSPipeline] Pipeline shut down cleanly." << std::endl;
    }
}

void RK3588BCSPipeline::decode_stage() {
    while (running_) {
        ZeroCopyFrame frame;
        if (!decoder_->decode_next_frame(frame)) break;

        last_frame_timestamp_ms_ = frame.timestamp_ms;

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            decode_to_rga_queue_.push(frame);
        }
        cv_.notify_all();

        if (frame.frame_id >= 1000) break;

        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / current_fps_limit_.load()));
    }
}

void RK3588BCSPipeline::rga_stage() {
    while (running_) {
        ZeroCopyFrame frame;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { return !decode_to_rga_queue_.empty() || !running_; });
            if (!running_ && decode_to_rga_queue_.empty()) break;

            frame = decode_to_rga_queue_.front();
            decode_to_rga_queue_.pop();
        }

        CropRegion crop{300, 150, 1350, 880};
        int out_dma_buf_fd;
        rga_resizer_->crop_and_resize_zerocopy(frame, crop, 224, 224, out_dma_buf_fd);

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            rga_to_npu_queue_.push(frame);
        }
        cv_.notify_all();
    }
}

void RK3588BCSPipeline::npu_stage() {
    while (running_) {
        ZeroCopyFrame frame;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { return !rga_to_npu_queue_.empty() || !running_; });
            if (!running_ && rga_to_npu_queue_.empty()) break;

            frame = rga_to_npu_queue_.front();
            rga_to_npu_queue_.pop();
        }

        std::vector<RKNNBox> detections;
        yolo_npu_->run_yolo_dma(frame.mpp_dma_buf_fd, detections);

        for (const auto& det : detections) {
            std::vector<float> embedding;
            dinov2_npu_->run_dinov2_dma(frame.mpp_dma_buf_fd, embedding);

            ClassificationResult res = classifier_->predict(embedding);

            if (frame.frame_id % 100 == 0) {
                std::cout << "[RK3588 Frame " << frame.frame_id << "] Cow detected (Conf: "
                          << det.confidence << ") -> BCS Score: " << res.label
                          << " (Prob: " << res.confidence << ")" << std::endl;
            }
        }
    }
}

void RK3588BCSPipeline::thermal_monitor() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        int temp = 55;
        if (temp > 75 && current_fps_limit_ > 12) {
            current_fps_limit_ = 12;
        } else if (temp < 65 && current_fps_limit_ < 25) {
            current_fps_limit_ = 25;
        }
    }
}

void RK3588BCSPipeline::watchdog_daemon() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

} // namespace rk3588
