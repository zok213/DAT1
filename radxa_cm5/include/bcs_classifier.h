#ifndef RK3588_BCS_CLASSIFIER_H
#define RK3588_BCS_CLASSIFIER_H

#include <vector>
#include <string>
#include <cmath>

namespace rk3588 {

struct ClassificationResult {
    int class_id;
    std::string label;
    float confidence;
    std::vector<float> logits;
};

class BCSClassifier {
public:
    BCSClassifier(int in_dim = 384, int d_model = 128, int n_classes = 3);
    ~BCSClassifier() = default;

    bool load_weights(const std::string& weights_path);
    ClassificationResult predict(const std::vector<float>& embedding);
    ClassificationResult predict_smoothed(const std::vector<float>& embedding, float alpha = 0.25f);
    void reset_smoothing();

private:
    int in_dim_;
    int d_model_;
    int n_classes_;
    std::vector<float> prev_probs_;

    std::vector<float> w_proj_, b_proj_;
    std::vector<float> w_head_, b_head_;
    std::vector<float> w_cls_, b_cls_;

    std::vector<float> layer_norm(const std::vector<float>& input);
    std::vector<float> gelu(const std::vector<float>& input);
    std::vector<float> matmul(const std::vector<float>& input, const std::vector<float>& weights, const std::vector<float>& bias, int in_d, int out_d);
    std::vector<float> softmax(const std::vector<float>& logits);
};

} // namespace rk3588

#endif // RK3588_BCS_CLASSIFIER_H
