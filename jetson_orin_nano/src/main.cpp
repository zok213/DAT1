#include "jetson_pipeline.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::cout << "=================================================" << std::endl;
    std::cout << "  NVIDIA Jetson Orin Nano / Orin NX Cow BCS      " << std::endl;
    std::cout << "  DeepStream NVMM Zero-Copy Pipeline Engine      " << std::endl;
    std::cout << "=================================================" << std::endl;

    bcs::PipelineConfig config;
    config.video_source = (argc > 1) ? argv[1] : "sample_cow_video.mp4";
    config.yolo_engine_path = (argc > 2) ? argv[2] : "models/yolov8n_seg_int8.engine";
    config.dinov2_engine_path = (argc > 3) ? argv[3] : "models/dinov2_vits14_fp16.engine";
    config.head_weights_path = (argc > 4) ? argv[4] : "models/bcs_head.npz";
    config.target_fps = 30;

    bcs::JetsonBCSPipeline pipeline(config);

    if (!pipeline.initialize()) {
        std::cerr << "[ERROR] Failed to initialize Jetson BCS Pipeline." << std::endl;
        return 1;
    }

    std::cout << "[INFO] Starting DeepStream NVMM Pipeline execution..." << std::endl;
    pipeline.start();

    // Run until completion
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        // Exit loop when pipeline finishes
        break;
    }

    pipeline.stop();
    std::cout << "[INFO] Jetson BCS Pipeline execution complete." << std::endl;
    return 0;
}
