#ifndef JETSON_PIPELINE_H
#define JETSON_PIPELINE_H

#include "nvmm_decoder.h"
#include "tensorrt_engine.h"
#include "bcs_classifier.h"

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <memory>

namespace bcs {

struct PipelineConfig {
    std::string video_source;
    std::string yolo_engine_path;
    std::string dinov2_engine_path;
    std::string head_weights_path;
    int target_fps = 30;
    bool enable_thermal_throttling = true;
    bool enable_watchdog = true;
};

class JetsonBCSPipeline {
public:
    JetsonBCSPipeline(const PipelineConfig& config);
    ~JetsonBCSPipeline();

    bool initialize();
    void start();
    void stop();

private:
    PipelineConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<int> current_fps_limit_{30};
    std::atomic<int64_t> last_frame_timestamp_ms_{0};

    std::unique_ptr<NVMMDecoder> decoder_;
    std::unique_ptr<TensorRTEngine> yolo_engine_;
    std::unique_ptr<TensorRTEngine> dinov2_engine_;
    std::unique_ptr<BCSClassifier> classifier_;

    std::thread pipeline_thread_;
    std::thread thermal_thread_;
    std::thread watchdog_thread_;

    void run_pipeline_loop();
    void run_thermal_monitor();
    void run_watchdog();
};

} // namespace bcs

#endif // JETSON_PIPELINE_H
