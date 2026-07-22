#include "rk3588_pipeline.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::cout << "=================================================" << std::endl;
    std::cout << "  Radxa CM5 (Rockchip RK3588) Cow BCS Pipeline   " << std::endl;
    std::cout << "  Zero-Copy MPP + RGA + RKNN NPU 3-Core Engine   " << std::endl;
    std::cout << "=================================================" << std::endl;

    rk3588::PipelineConfig config;
    config.video_source = (argc > 1) ? argv[1] : "sample_cow_video.mp4";
    config.yolo_rknn_path = (argc > 2) ? argv[2] : "models/yolov8n_seg.rknn";
    config.dinov2_rknn_path = (argc > 3) ? argv[3] : "models/dinov2_vits14.rknn";
    config.head_weights_path = (argc > 4) ? argv[4] : "models/bcs_head.npz";
    config.target_fps = 25;

    rk3588::RK3588BCSPipeline pipeline(config);

    if (!pipeline.initialize()) {
        std::cerr << "[ERROR] Failed to initialize RK3588 BCS Pipeline." << std::endl;
        return 1;
    }

    std::cout << "[INFO] Starting RK3588 Zero-Copy Pipeline..." << std::endl;
    pipeline.start();

    // Run until completion
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        break;
    }

    pipeline.stop();
    std::cout << "[INFO] RK3588 BCS Pipeline execution complete." << std::endl;
    return 0;
}
