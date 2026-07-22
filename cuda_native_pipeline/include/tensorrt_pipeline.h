#ifndef TENSORRT_PIPELINE_H
#define TENSORRT_PIPELINE_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "cuda_kernels.h"

struct BCSResult {
    int class_id;
    std::string label;
    float confidence;
    float probabilities[3];
    float total_ms;
};

class CudaBCSPipeline {
public:
    CudaBCSPipeline(const std::string& yolo_model, const std::string& dino_model, const std::string& head_model);
    ~CudaBCSPipeline();

    BCSResult process_frame(const cv::Mat& frame_bgr);

private:
    cudaStream_t stream_;
    uint8_t* d_frame_bgr_;
    float* d_dino_input_;
    float ema_state_[3];
    bool is_ema_init_;
    float alpha_;
};

#endif // TENSORRT_PIPELINE_H
