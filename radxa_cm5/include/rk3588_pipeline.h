#ifndef RK3588_PIPELINE_H
#define RK3588_PIPELINE_H

#include "mpp_decoder.h"
#include "rga_resizer.h"
#include "rknn_engine.h"
#include "bcs_classifier.h"

#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <memory>

namespace rk3588 {

struct PipelineConfig {
    std::string video_source;
    std::string yolo_rknn_path;
    std::string dinov2_rknn_path;
    std::string head_weights_path;
    int target_fps = 25;
    bool enable_thermal_throttling = true;
    bool enable_watchdog = true;
};

class RK3588BCSPipeline {
public:
    RK3588BCSPipeline(const PipelineConfig& config);
    ~RK3588BCSPipeline();

    bool initialize();
    void start();
    void stop();

private:
    PipelineConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<int> current_fps_limit_{25};
    std::atomic<int64_t> last_frame_timestamp_ms_{0};

    std::unique_ptr<MPPDecoder> decoder_;
    std::unique_ptr<RGAResizer> rga_resizer_;
    std::unique_ptr<RKNNEngine> yolo_npu_;
    std::unique_ptr<RKNNEngine> dinov2_npu_;
    std::unique_ptr<BCSClassifier> classifier_;

    std::queue<ZeroCopyFrame> decode_to_rga_queue_;
    std::queue<ZeroCopyFrame> rga_to_npu_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;

    std::thread decode_thread_;
    std::thread rga_thread_;
    std::thread npu_thread_;
    std::thread thermal_thread_;
    std::thread watchdog_thread_;

    void decode_stage();
    void rga_stage();
    void npu_stage();
    void thermal_monitor();
    void watchdog_daemon();
};

} // namespace rk3588

#endif // RK3588_PIPELINE_H
