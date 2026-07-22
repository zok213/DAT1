#ifndef TENSORRT_ENGINE_H
#define TENSORRT_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace bcs {

struct DetectionBox {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
    std::vector<float> mask;
};

class TensorRTEngine {
public:
    TensorRTEngine(const std::string& engine_path, bool is_fp16 = true);
    ~TensorRTEngine();

    bool load_engine();
    bool execute_yolo(void* nvmm_input_ptr, std::vector<DetectionBox>& detections);
    bool execute_dinov2(const std::vector<float>& cropped_tensor_224, std::vector<float>& embedding_384);

private:
    std::string engine_path_;
    bool is_fp16_;
    bool engine_loaded_;
    void* context_handle_;
};

} // namespace bcs

#endif // TENSORRT_ENGINE_H
