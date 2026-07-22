#ifndef RGA_RESIZER_H
#define RGA_RESIZER_H

#include "mpp_decoder.h"
#include <cstdint>

namespace rk3588 {

struct CropRegion {
    int x1, y1, x2, y2;
};

class RGAResizer {
public:
    RGAResizer();
    ~RGAResizer();

    bool initialize();
    bool crop_and_resize_zerocopy(const ZeroCopyFrame& in_frame, const CropRegion& crop, int target_w, int target_h, int& out_dma_buf_fd);

private:
    void* rga_ctx_;
};

} // namespace rk3588

#endif // RGA_RESIZER_H
