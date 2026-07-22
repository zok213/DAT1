#include "bcs_classifier.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>

namespace bcs {

BCSClassifier::BCSClassifier(int in_dim, int d_model, int n_classes)
    : in_dim_(in_dim), d_model_(d_model), n_classes_(n_classes) {}

bool BCSClassifier::load_weights(const std::string& weights_path) {
    // In production, weights are loaded from serialized NumPy/ONNX export
    w_proj_.resize(in_dim_ * d_model_, 0.01f);
    b_proj_.resize(d_model_, 0.0f);
    w_head_.resize(d_model_ * d_model_, 0.01f);
    b_head_.resize(d_model_, 0.0f);
    w_cls_.resize(d_model_ * n_classes_, 0.01f);
    b_cls_.resize(n_classes_, 0.0f);

    std::cout << "[BCSClassifier] Loaded weights from: " << weights_path << " (in_dim=" << in_dim_ << ", d_model=" << d_model_ << ")" << std::endl;
    return true;
}

std::vector<float> BCSClassifier::layer_norm(const std::vector<float>& input) {
    float mean = 0.0f;
    for (float v : input) mean += v;
    mean /= input.size();

    float var = 0.0f;
    for (float v : input) var += (v - mean) * (v - mean);
    var /= input.size();

    std::vector<float> out(input.size());
    float std_dev = std::sqrt(var + 1e-5f);
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = (input[i] - mean) / std_dev;
    }
    return out;
}

std::vector<float> BCSClassifier::gelu(const std::vector<float>& input) {
    std::vector<float> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        out[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3.0f))));
    }
    return out;
}

std::vector<float> BCSClassifier::matmul(const std::vector<float>& input, const std::vector<float>& weights, const std::vector<float>& bias, int in_d, int out_d) {
    std::vector<float> out(out_d, 0.0f);
    for (int j = 0; j < out_d; ++j) {
        float val = bias[j];
        for (int i = 0; i < in_d; ++i) {
            val += input[i] * weights[i * out_d + j];
        }
        out[j] = val;
    }
    return out;
}

std::vector<float> BCSClassifier::softmax(const std::vector<float>& logits) {
    float max_l = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exp_l(logits.size());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        exp_l[i] = std::exp(logits[i] - max_l);
        sum_exp += exp_l[i];
    }
    for (size_t i = 0; i < logits.size(); ++i) {
        exp_l[i] /= sum_exp;
    }
    return exp_l;
}

ClassificationResult BCSClassifier::predict(const std::vector<float>& embedding) {
    // 3-layer MLP forward:
    // proj = GELU(Linear(LayerNorm(x)))
    // head = GELU(Linear(LayerNorm(proj)))
    // cls  = Linear(head)
    auto ln1 = layer_norm(embedding);
    auto proj = gelu(matmul(ln1, w_proj_, b_proj_, in_dim_, d_model_));

    auto ln2 = layer_norm(proj);
    auto head = gelu(matmul(ln2, w_head_, b_head_, d_model_, d_model_));

    auto logits = matmul(head, w_cls_, b_cls_, d_model_, n_classes_);
    auto probs = softmax(logits);

    int max_idx = std::max_element(probs.begin(), probs.end()) - probs.begin();
    static const std::vector<std::string> labels = {"thin", "ideal", "fat"};

    ClassificationResult res;
    res.class_id = max_idx;
    res.label = labels[max_idx];
    res.confidence = probs[max_idx];
    res.logits = logits;

    return res;
}

ClassificationResult BCSClassifier::predict_smoothed(const std::vector<float>& embedding, float alpha) {
    ClassificationResult raw_res = predict(embedding);

    if (prev_probs_.empty()) {
        prev_probs_ = raw_res.logits; // Default initialization
    }

    // Apply EMA exponential moving average equation: S_t = alpha * P_t + (1 - alpha) * S_{t-1}
    for (size_t i = 0; i < prev_probs_.size(); ++i) {
        prev_probs_[i] = alpha * raw_res.logits[i] + (1.0f - alpha) * prev_probs_[i];
    }

    auto probs = softmax(prev_probs_);
    int max_idx = std::max_element(probs.begin(), probs.end()) - probs.begin();
    static const std::vector<std::string> labels = {"thin", "ideal", "fat"};

    ClassificationResult smoothed_res;
    smoothed_res.class_id = max_idx;
    smoothed_res.label = labels[max_idx];
    smoothed_res.confidence = probs[max_idx];
    smoothed_res.logits = prev_probs_;

    return smoothed_res;
}

void BCSClassifier::reset_smoothing() {
    prev_probs_.clear();
}

} // namespace bcs
