#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <cstdint>

// CUDA Kernel: Crops ROI bounding box, multiplies segmentation mask, resizes to 224x224, and normalizes (ImageNet mean/std)
extern "C" void launch_crop_mask_normalize_kernel(
    const uint8_t* d_src_bgr,
    int src_width,
    int src_height,
    int crop_x1,
    int crop_y1,
    int crop_x2,
    int crop_y2,
    const uint8_t* d_mask,
    float* d_dst_tensor, // (1, 3, 224, 224) normalized float32
    cudaStream_t stream
);

#endif // CUDA_KERNELS_H
