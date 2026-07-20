#include <iostream>
// #include <onnxruntime_cxx_api.h>

class OnnxRunner {
public:
    OnnxRunner(const std::string& model_path) {
        std::cout << "[ONNX] Loading model from: " << model_path << std::endl;
        // env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "cowdeploy");
        // session_options.SetIntraOpNumThreads(4);
        // session = Ort::Session(env, model_path.c_str(), session_options);
    }

    void warmup() {
        std::cout << "[ONNX] Running warmup iterations..." << std::endl;
    }

    void benchmark() {
        std::cout << "[ONNX] Benchmarking ONNX inference..." << std::endl;
    }
};
