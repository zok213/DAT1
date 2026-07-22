#ifndef MPP_DECODER_H
#define MPP_DECODER_H

#include <string>
#include <cstdint>

namespace rk3588 {

struct ZeroCopyFrame {
    uint32_t frame_id;
    int mpp_dma_buf_fd; // File descriptor for hardware memory block
    uint32_t width;
    uint32_t height;
    uint64_t timestamp_ms;
};

class MPPDecoder {
public:
    MPPDecoder(const std::string& video_path);
    ~MPPDecoder();

    bool initialize();
    bool decode_next_frame(ZeroCopyFrame& frame);
    void release();

private:
    std::string video_path_;
    uint32_t frame_counter_;
    void* mpp_ctx_;
};

} // namespace rk3588

#endif // MPP_DECODER_H
