#include "cuda_kernels.h"
#include <cuda_fp16.h>

__global__ void crop_mask_normalize_kernel(
    const uint8_t* __restrict__ src,
    int src_w, int src_h,
    int x1, int y1, int x2, int y2,
    const uint8_t* __restrict__ mask,
    float* __restrict__ dst
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x; // [0..223]
    int out_y = blockIdx.y * blockDim.y + threadIdx.y; // [0..223]

    if (out_x >= 224 || out_y >= 224) return;

    int crop_w = x2 - x1;
    int crop_h = y2 - y1;

    // Bilinear Mapping coordinates back to source frame
    float src_x_f = x1 + (out_x + 0.5f) * ((float)crop_w / 224.0f) - 0.5f;
    float src_y_f = y1 + (out_y + 0.5f) * ((float)crop_h / 224.0f) - 0.5f;

    int src_x = min(max((int)src_x_f, 0), src_w - 1);
    int src_y = min(max((int)src_y_f, 0), src_h - 1);

    int src_idx = (src_y * src_w + src_x) * 3;

    // Mask check
    uint8_t m_val = 1;
    if (mask != nullptr) {
        m_val = mask[src_y * src_w + src_x] > 0 ? 1 : 0;
    }

    // BGR -> RGB normalized float32
    float b = (float)src[src_idx + 0] * m_val / 255.0f;
    float g = (float)src[src_idx + 1] * m_val / 255.0f;
    float r = (float)src[src_idx + 2] * m_val / 255.0f;

    // ImageNet mean/std normalization
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3]  = {0.229f, 0.224f, 0.225f};

    r = (r - mean[0]) / std[0];
    g = (g - mean[1]) / std[1];
    b = (b - mean[2]) / std[2];

    int plane_stride = 224 * 224;
    int dst_offset = out_y * 224 + out_x;

    // Planar CHW format: [R_plane, G_plane, B_plane]
    dst[0 * plane_stride + dst_offset] = r;
    dst[1 * plane_stride + dst_offset] = g;
    dst[2 * plane_stride + dst_offset] = b;
}

extern "C" void launch_crop_mask_normalize_kernel(
    const uint8_t* d_src_bgr,
    int src_width,
    int src_height,
    int crop_x1,
    int crop_y1,
    int crop_x2,
    int crop_y2,
    const uint8_t* d_mask,
    float* d_dst_tensor,
    cudaStream_t stream
) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((224 + 15) / 16, (224 + 15) / 16);

    crop_mask_normalize_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_src_bgr, src_width, src_height,
        crop_x1, crop_y1, crop_x2, crop_y2,
        d_mask, d_dst_tensor
    );
}
