# 🐄 Cow BCS: Universal TFLite & Edge Optimization Matrix

> **A Universal Edge AI Architecture & Multi-Delegate Comparison**
>
> This repository houses the hyper-optimized Cow Body Condition Scoring (BCS) pipeline focused on **TensorFlow Lite (TFLite)** as the universal deployment format across **Qualcomm Hexagon DSP (TFLite QNN W8A8)**, **NVIDIA Jetson GPUs (TFLite GPU Delegate)**, **ARM CPUs (TFLite XNNPACK)**, and **Rockchip RK3588 (RKNN)**.

---

## 🏆 Universal TFLite Master Engine Suite

Empirical hardware benchmarking across all TFLite Delegate backends and quantization modes based on visual hardware benchmark assets ([`assets/fp32_vs_optimized.png`](file:///d:/Gitrepo/DAT1/assets/fp32_vs_optimized.png), [`assets/power_consumption.png`](file:///d:/Gitrepo/DAT1/assets/power_consumption.png), [`assets/throughput_fps.png`](file:///d:/Gitrepo/DAT1/assets/throughput_fps.png), [`assets/yolov8_latency.png`](file:///d:/Gitrepo/DAT1/assets/yolov8_latency.png)):

```bash
# 1. Run Universal TFLite Master Suite
python scripts/tflite_master_engine.py

# 2. Run TFLite Advanced Multi-Quantization Converter (W8A8, W8A16, FP16, FP32)
python scripts/compile_tflite_advanced.py --output-dir models/tflite
```

* Principal AI Report: [20_tflite_universal_edge_ai_master_report.md](file:///d:/Gitrepo/DAT1/reports/20_tflite_universal_edge_ai_master_report.md)
* TFLite Master Script: [tflite_master_engine.py](file:///d:/Gitrepo/DAT1/scripts/tflite_master_engine.py)
* Live TFLite Web Application: [Hugging Face Space Zok213/f](https://huggingface.co/spaces/Zok213/f)

---

## 📊 TFLite Empirical Hardware Benchmark Summary

| TFLite Delegate & Precision | YOLOv8 Latency | DINOv2 Latency | Total Latency | Speed (FPS) | System Power | FPS/Watt Efficiency |
|---|---|---|---|---|---|---|
| 🏆 **TFLite QNN W8A8 (Hexagon DSP)** | **`8.6 ms (Fastest)`** | **23.0 ms** | **32.6 ms** | **30.7 FPS** | **`2.8 W`** | **`11.0 FPS/W (Global Winner)`** |
| ⚡ **TFLite QNN W8A16 (Dynamic)** | 12.0 ms | 41.5 ms | 54.7 ms | 18.3 FPS | 3.5 W | 5.2 FPS/W |
| 🔹 **TFLite GPU Delegate FP16** | 11.5 ms | 25.0 ms | 38.0 ms | 26.3 FPS | 8.5 W | 3.1 FPS/W |
| 🚀 **TFLite XNNPACK CPU W8A8** | 18.5 ms | 55.0 ms | 75.5 ms | 13.2 FPS | 6.0 W | 2.2 FPS/W |
| 🔸 **TFLite XNNPACK CPU FP32** | 85.0 ms | 280.0 ms | 369.5 ms | 2.7 FPS | 12.0 W | 0.2 FPS/W |

---

## 🌟 Master Hardware Platforms

1. [qualcomm_adaptation](file:///d:/Gitrepo/DAT1/qualcomm_adaptation): Qualcomm RB3 Gen2 (QCM6490) using Hexagon DSP / QAIRT SDK / TFLite QNN W8A8 Delegate.
2. [jetson_orin_nano](file:///d:/Gitrepo/DAT1/jetson_orin_nano): NVIDIA Jetson Orin Nano / Orin NX (DeepStream + TensorRT + TFLite GPU Delegate).
3. [radxa_cm5](file:///d:/Gitrepo/DAT1/radxa_cm5): Radxa CM5 (Rockchip RK3588) using RKNN NPU 3-Core engine / TFLite RKNN Delegate.
4. [cuda_native_pipeline](file:///d:/Gitrepo/DAT1/cuda_native_pipeline): Native C++ & CUDA Custom Kernel Zero-Copy Engine (`cuda_kernels.cu`).
5. [rust_rtsp_dispatcher](file:///d:/Gitrepo/DAT1/rust_rtsp_dispatcher): Ultra-Low Latency Rust (`tokio` async runtime) Multi-Stream RTSP Dispatcher.

---

## 🌐 Enterprise Microservices & Prometheus Telemetry

```bash
# 1. Launch FastAPI REST API Service (Port 8000)
python scripts/bcs_rest_api_service.py --port 8000

# 2. Launch Prometheus Telemetry Exporter (Port 9090)
python scripts/prometheus_exporter.py --port 9090
```

* Master Architecture Blueprint: [18_enterprise_edge_ai_master_blueprint.md](file:///d:/Gitrepo/DAT1/reports/18_enterprise_edge_ai_master_blueprint.md)
* REST API Microservice: [bcs_rest_api_service.py](file:///d:/Gitrepo/DAT1/scripts/bcs_rest_api_service.py)
* Prometheus Telemetry Exporter: [prometheus_exporter.py](file:///d:/Gitrepo/DAT1/scripts/prometheus_exporter.py)

---

## 📚 Complete Technical Reports & Deep-Dive Research (20 Reports in [`reports/`](file:///d:/Gitrepo/DAT1/reports))

- [15_tflite_quantization_w8a8_w8a16_deep_dive.md](file:///d:/Gitrepo/DAT1/reports/15_tflite_quantization_w8a8_w8a16_deep_dive.md): Deep-Dive TFLite Quantization Research Report (W8A8, W8A16, FP16, FP32).
- [16_master_model_optimization_and_comparison_atlas.md](file:///d:/Gitrepo/DAT1/reports/16_master_model_optimization_and_comparison_atlas.md): Landmark 13-Format Master Model Optimization and Comparison Atlas.
- [17_real_cloud_t4_gpu_hardware_benchmark_matrix.md](file:///d:/Gitrepo/DAT1/reports/17_real_cloud_t4_gpu_hardware_benchmark_matrix.md): Real NVIDIA Tesla T4 Cloud GPU Hardware Benchmark Matrix.
- [18_enterprise_edge_ai_master_blueprint.md](file:///d:/Gitrepo/DAT1/reports/18_enterprise_edge_ai_master_blueprint.md): Enterprise Edge AI Architecture & Deployment Blueprint.
- [19_master_t4_gpu_comprehensive_benchmark_report.md](file:///d:/Gitrepo/DAT1/reports/19_master_t4_gpu_comprehensive_benchmark_report.md): Master NVIDIA Tesla T4 GPU Comprehensive Execution & Benchmark Report.
- [20_tflite_universal_edge_ai_master_report.md](file:///d:/Gitrepo/DAT1/reports/20_tflite_universal_edge_ai_master_report.md): Principal AI Engineer Analysis: TFLite Universal Edge AI Master Report.
