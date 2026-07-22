#ifndef RKNN_ENGINE_H
#define RKNN_ENGINE_H

#include <string>
#include <vector>
#include <cstdint>

namespace rk3588 {

struct RKNNBox {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

class RKNNEngine {
public:
    RKNNEngine(const std::string& model_path);
    ~RKNNEngine();

    bool load_model();
    bool run_yolo_dma(int dma_buf_fd, std::vector<RKNNBox>& detections);
    bool run_dinov2_dma(int dma_buf_fd, std::vector<float>& embedding_384);

private:
    std::string model_path_;
    uint64_t rknn_ctx_;
    bool loaded_;
};

} // namespace rk3588

#endif // RKNN_ENGINE_H
