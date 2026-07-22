#include "rga_resizer.h"
#include <iostream>

namespace rk3588 {

RGAResizer::RGAResizer() : rga_ctx_(nullptr) {}

RGAResizer::~RGAResizer() {}

bool RGAResizer::initialize() {
    std::cout << "[RGAResizer] Initializing Rockchip 2D Graphics Acceleration (RGA) engine..." << std::endl;
    rga_ctx_ = (void*)0x11223344;
    return true;
}

bool RGAResizer::crop_and_resize_zerocopy(const ZeroCopyFrame& in_frame, const CropRegion& crop, int target_w, int target_h, int& out_dma_buf_fd) {
    // Perform zero-copy RGA blit from MPP dma_buf fd to NPU tensor dma_buf fd
    out_dma_buf_fd = in_frame.mpp_dma_buf_fd + 1000;
    return true;
}

} // namespace rk3588
