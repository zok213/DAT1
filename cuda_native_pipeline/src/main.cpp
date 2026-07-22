#include <iostream>
#include <iomanip>
#include "tensorrt_pipeline.h"

int main(int argc, char** argv) {
    std::cout << "=================================================" << std::endl;
    std::cout << " C++ CUDA Native Zero-Copy BCS Pipeline Engine  " << std::endl;
    std::cout << " Custom CUDA Kernels + TensorRT Execution Suite " << std::endl;
    std::cout << "=================================================" << std::endl;

    std::string video_path = (argc > 1) ? argv[1] : "sample_cow_video.mp4";

    CudaBCSPipeline pipeline("yolov8n-seg.engine", "dinov2_vits14.engine", "bcs_head.engine");

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cout << "[INFO] Test synthetic frame execution mode..." << std::endl;
        cv::Mat dummy_frame(1080, 1920, CV_8UC3, cv::Scalar(128, 128, 128));

        float total_time_ms = 0.0f;
        int benchmark_iterations = 200;

        for (int i = 1; i <= benchmark_iterations; ++i) {
            BCSResult res = pipeline.process_frame(dummy_frame);
            total_time_ms += res.total_ms;

            if (i % 50 == 0 || i == 1) {
                std::cout << "[CUDA C++] Frame #" << std::setw(3) << i 
                          << " | Pred: " << std::setw(6) << res.label 
                          << " (" << std::fixed << std::setprecision(1) << (res.confidence * 100.0f) << "%)"
                          << " | Latency: " << std::setprecision(2) << res.total_ms << "ms" 
                          << " | FPS: " << std::setprecision(1) << (1000.0f / res.total_ms) << std::endl;
            }
        }

        float avg_ms = total_time_ms / benchmark_iterations;
        std::cout << "\n=================================================" << std::endl;
        std::cout << " [SUCCESS] Benchmark Complete!" << std::endl;
        std::cout << " Average Frame Latency : " << avg_ms << " ms" << std::endl;
        std::cout << " Achieved Throughput   : " << (1000.0f / avg_ms) << " FPS" << std::endl;
        std::cout << "=================================================" << std::endl;
        return 0;
    }

    cv::Mat frame;
    int frame_id = 0;
    while (cap.read(frame)) {
        frame_id++;
        BCSResult res = pipeline.process_frame(frame);
        if (frame_id % 30 == 0) {
            std::cout << "[CUDA C++] Video Frame #" << frame_id << " | BCS: " << res.label 
                      << " (" << (res.confidence * 100.0f) << "%) | Latency: " << res.total_ms << "ms" << std::endl;
        }
        if (frame_id >= 300) break;
    }

    return 0;
}
