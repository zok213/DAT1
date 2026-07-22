# 🐄 NVIDIA Jetson Orin Nano / Orin NX Deployment

This directory contains the production-grade C++ pipeline and inference runner optimized for NVIDIA Jetson Orin Nano and Orin NX platforms utilizing **DeepStream 6.x/7.x** and **NVMM Zero-Copy Memory**.

---

## ⚡ Key Optimizations

- **NVMM Zero-Copy Memory**: Uses `nvv4l2decoder` to decode H.264 video directly into GPU unified memory, avoiding CPU memory copies.
- **TensorRT FP16 / INT8**: YOLOv8n-seg and DINOv2 ViT-S/14 backbones are compiled into optimized `.engine` formats.
- **Enterprise Watchdog & Thermal Daemon**: Monitored real-time temperature throttling keeps hardware under thermal thresholds (throttles from 30 FPS to 15 FPS if temperature exceeds 75°C).

---

## 🏗️ C++ Source Architecture

- [`include/jetson_pipeline.h`](include/jetson_pipeline.h): High-level orchestrator interface.
- [`include/nvmm_decoder.h`](include/nvmm_decoder.h): NVIDIA NVMM hardware decoder interface.
- [`include/tensorrt_engine.h`](include/tensorrt_engine.h): TensorRT engine wrapper.
- [`include/bcs_classifier.h`](include/bcs_classifier.h): 3-Layer MLP BCS classifier head.
- [`src/main.cpp`](src/main.cpp): Executable entrypoint.

---

## 🚀 Building & Running

### 1. Build C++ Target
```bash
chmod +x scripts/build_jetson.sh
./scripts/build_jetson.sh
./build/jetson_cow_bcs sample_cow_video.mp4
```

### 2. Python Inference Runner
```bash
python3 scripts/run_jetson.py --video sample_cow_video.mp4
```
