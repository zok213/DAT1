#ifndef NVMM_DECODER_H
#define NVMM_DECODER_H

#include <string>
#include <memory>
#include <cstdint>

namespace bcs {

struct FrameMetadata {
    uint32_t frame_id;
    uint32_t width;
    uint32_t height;
    uint64_t timestamp_ms;
    void* nvmm_buffer_ptr; // Pointer to NVMM hardware buffer
};

class NVMMDecoder {
public:
    NVMMDecoder(const std::string& stream_uri);
    ~NVMMDecoder();

    bool initialize();
    bool read_frame(FrameMetadata& out_frame);
    void release();

    bool is_hardware_accelerated() const { return use_nvdec_; }

private:
    std::string uri_;
    bool use_nvdec_;
    uint32_t frame_counter_;
};

} // namespace bcs

#endif // NVMM_DECODER_H
