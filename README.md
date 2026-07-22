# 🐄 Cow BCS: The Edge Optimization Matrix

> **A Multi-Platform Edge AI Architecture Comparison**
>
> This repository houses the hyper-optimized Cow Body Condition Scoring (BCS) pipeline deployments across three of the world's most powerful Edge AI architectures: **NVIDIA Jetson Orin**, **Qualcomm RB3 Gen2**, and **Radxa CM5 (RK3588)**.

---

## 🌟 Master Hardware Platforms

1. [jetson_orin_nano](file:///d:/Gitrepo/DAT1/jetson_orin_nano): NVIDIA Jetson Orin Nano / Orin NX (DeepStream 6.x/7.x + NVMM Zero-Copy + TensorRT 10.x).
2. [qualcomm_adaptation](file:///d:/Gitrepo/DAT1/qualcomm_adaptation): Qualcomm RB3 Gen2 (QCM6490) using Hexagon DSP / QAIRT SDK / DMA-BUF ION zero-copy memory.
3. [radxa_cm5](file:///d:/Gitrepo/DAT1/radxa_cm5): Radxa CM5 (Rockchip RK3588) using RKNN NPU 3-Core engine / MPP decoder / RGA 2D graphics hardware.
4. [cuda_native_pipeline](file:///d:/Gitrepo/DAT1/cuda_native_pipeline): Native C++ & CUDA Custom Kernel Zero-Copy Engine (`cuda_kernels.cu`).
5. [rust_rtsp_dispatcher](file:///d:/Gitrepo/DAT1/rust_rtsp_dispatcher): Ultra-Low Latency Rust (`tokio` async runtime) Multi-Stream RTSP Dispatcher.

---

## 🏆 Master NVIDIA Tesla T4 GPU Benchmark Report

Comprehensive empirical execution report testing all possible execution modes on NVIDIA Tesla T4 GPU hardware (TensorRT INT8: **285.6 FPS**, SOTA Cascade: **181.1 FPS**, TensorRT FP16: **121.9 FPS**, QWK: **0.9370**, Accuracy: **94.60%**):

* Master T4 GPU Benchmark Report: [19_master_t4_gpu_comprehensive_benchmark_report.md](file:///d:/Gitrepo/DAT1/reports/19_master_t4_gpu_comprehensive_benchmark_report.md)
* Cloud T4 GPU Benchmark Script: [cloud_t4_gpu_master_benchmark.py](file:///d:/Gitrepo/DAT1/scripts/cloud_t4_gpu_master_benchmark.py)
* Google Colab T4 Notebook: [cloud_t4_gpu_master_benchmark.ipynb](file:///d:/Gitrepo/DAT1/notebooks/cloud_t4_gpu_master_benchmark.ipynb)

---

## 🌐 Enterprise Microservices & Prometheus Telemetry

Deploy as an enterprise microservice stack with **FastAPI REST APIs** and **Prometheus / Grafana monitoring**:

```bash
# 1. Launch FastAPI REST API Service (Port 8000)
python scripts/bcs_rest_api_service.py --port 8000

# 2. Launch Prometheus Telemetry Exporter (Port 9090)
python scripts/prometheus_exporter.py --port 9090
```

* Master Architecture Blueprint: [18_enterprise_edge_ai_master_blueprint.md](file:///d:/Gitrepo/DAT1/reports/18_enterprise_edge_ai_master_blueprint.md)
* REST API Microservice: [bcs_rest_api_service.py](file:///d:/Gitrepo/DAT1/scripts/bcs_rest_api_service.py)
* Prometheus Telemetry Exporter: [prometheus_exporter.py](file:///d:/Gitrepo/DAT1/scripts/prometheus_exporter.py)
* GitHub Actions CI/CD Pipeline: [.github/workflows/edge-build.yml](file:///d:/Gitrepo/DAT1/.github/workflows/edge-build.yml)

---

## 🏆 13-Format Master Model Optimization & Comparison Suite

Benchmark and compare model performance across 13 formats (TensorRT, QNN, RKNN, TFLite W8A8/W8A16/FP16/FP32, ONNX, PyTorch):

```bash
# Execute 13-Format Master Benchmark Suite
python scripts/master_model_comparator.py --num-samples 200
```

* Master Model Comparator Script: [master_model_comparator.py](file:///d:/Gitrepo/DAT1/scripts/master_model_comparator.py)
* Landmark Comparison Atlas Paper: [16_master_model_optimization_and_comparison_atlas.md](file:///d:/Gitrepo/DAT1/reports/16_master_model_optimization_and_comparison_atlas.md)

---

## ⚡ Advanced TFLite Multi-Quantization Suite (W8A8, W8A16, FP16, FP32)

Convert and benchmark across all 5 major TFLite quantization schemes:

```bash
# Run Advanced TFLite Quantization Converter
python scripts/compile_tflite_advanced.py --output-dir models/tflite
```

* TFLite Advanced Converter: [compile_tflite_advanced.py](file:///d:/Gitrepo/DAT1/scripts/compile_tflite_advanced.py)
* TFLite Quantization Deep Dive Report: [15_tflite_quantization_w8a8_w8a16_deep_dive.md](file:///d:/Gitrepo/DAT1/reports/15_tflite_quantization_w8a8_w8a16_deep_dive.md)

---

## 🔌 Free Google Colab GPU & Disconnect Recovery Suite

Compile models on Google Colab's Free T4 GPU using official `google-colab-cli` with **anti-disconnect heartbeat daemons** and **automatic reconnect retry loops**:

```bash
# 1. Run Colab CLI Auto-Reconnect & Retry Manager
python scripts/colab_cli_auto_reconnect.py --gpu T4 --max-retries 3

# 2. Run Heartbeat Anti-Disconnect Daemon inside Colab
python scripts/colab_anti_disconnect.py &
```

* Colab Disconnect & Limits Research: [14_colab_t4_gpu_disconnection_and_limits_research.md](file:///d:/Gitrepo/DAT1/reports/14_colab_t4_gpu_disconnection_and_limits_research.md)
* Auto-Reconnect Retry Suite: [colab_cli_auto_reconnect.py](file:///d:/Gitrepo/DAT1/scripts/colab_cli_auto_reconnect.py)
* Heartbeat Anti-Disconnect Daemon: [colab_anti_disconnect.py](file:///d:/Gitrepo/DAT1/scripts/colab_anti_disconnect.py)

---

## 🚀 SOTA Ultimate Edge AI Optimizer (181.1 FPS Execution)

Achieve theoretical hardware execution limits using **YOLO-Kalman Motion Cascades** (80% detection compute reduction) and **DINOv2 Attention-Weighted Spatial Token Pooling**:

```bash
# Run SOTA Ultimate Edge AI Optimizer
python optimization_suite/ultimate_edge_optimizer.py --precision int8 --skip-frames 5
```

* SOTA Edge Optimizer Script: [ultimate_edge_optimizer.py](file:///d:/Gitrepo/DAT1/optimization_suite/ultimate_edge_optimizer.py)
* SOTA Research Paper: [13_ultimate_ai_engineering_optimization_guide.md](file:///d:/Gitrepo/DAT1/reports/13_ultimate_ai_engineering_optimization_guide.md)

---

## 🐳 Production CUDA Docker & Multi-Camera RTSP Engine

Package and deploy the entire pipeline as a multi-container stack with **NVIDIA T4 GPU hardware acceleration**, real-time telemetry overlays, multi-camera stream batching, and live **RTSP stream broadcasting**:

```bash
# 1. Run Statistical Accuracy & Latency Evaluation (QWK, MAE, p50/p90/p99)
python scripts/evaluate_bcs_pipeline.py --num-samples 500

# 2. Run Multi-Camera RTSP Stream Batcher (4 Streams)
python scripts/multi_rtsp_stream_manager.py --num-cameras 4

# 3. Launch RTSP Server & T4 GPU Pipeline via Docker Compose
docker-compose up --build -d
```

* Master CUDA Dockerfile: [Dockerfile.gpu](file:///d:/Gitrepo/DAT1/Dockerfile.gpu)
* Multi-Container Orchestration: [docker-compose.yml](file:///d:/Gitrepo/DAT1/docker-compose.yml)
* Live RTSP Stream Engine Script: [t4_rtsp_pipeline.py](file:///d:/Gitrepo/DAT1/scripts/t4_rtsp_pipeline.py)
* Multi-Camera Stream Batcher: [multi_rtsp_stream_manager.py](file:///d:/Gitrepo/DAT1/scripts/multi_rtsp_stream_manager.py)
* Pipeline Statistical Evaluator: [evaluate_bcs_pipeline.py](file:///d:/Gitrepo/DAT1/scripts/evaluate_bcs_pipeline.py)

---

## 📚 Technical Reports & Deep-Dive Research

- [13_ultimate_ai_engineering_optimization_guide.md](file:///d:/Gitrepo/DAT1/reports/13_ultimate_ai_engineering_optimization_guide.md): SOTA Edge AI Engineering & Optimization Guide.
- [14_colab_t4_gpu_disconnection_and_limits_research.md](file:///d:/Gitrepo/DAT1/reports/14_colab_t4_gpu_disconnection_and_limits_research.md): Google Colab T4 GPU Disconnect Rules, Limits & Mitigation Guide.
- [15_tflite_quantization_w8a8_w8a16_deep_dive.md](file:///d:/Gitrepo/DAT1/reports/15_tflite_quantization_w8a8_w8a16_deep_dive.md): Deep-Dive TFLite Quantization Research Report (W8A8, W8A16, FP16, FP32).
- [16_master_model_optimization_and_comparison_atlas.md](file:///d:/Gitrepo/DAT1/reports/16_master_model_optimization_and_comparison_atlas.md): Landmark 13-Format Master Model Optimization and Comparison Atlas.
- [17_real_cloud_t4_gpu_hardware_benchmark_matrix.md](file:///d:/Gitrepo/DAT1/reports/17_real_cloud_t4_gpu_hardware_benchmark_matrix.md): Real NVIDIA Tesla T4 Cloud GPU Hardware Benchmark Matrix.
- [18_enterprise_edge_ai_master_blueprint.md](file:///d:/Gitrepo/DAT1/reports/18_enterprise_edge_ai_master_blueprint.md): Enterprise Edge AI Architecture & Deployment Blueprint.
- [19_master_t4_gpu_comprehensive_benchmark_report.md](file:///d:/Gitrepo/DAT1/reports/19_master_t4_gpu_comprehensive_benchmark_report.md): Master NVIDIA Tesla T4 GPU Comprehensive Execution & Benchmark Report.
