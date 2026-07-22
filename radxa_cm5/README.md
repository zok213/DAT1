# 🐄 Radxa CM5 (Rockchip RK3588) Deployment

This directory contains the production-grade C++ pipeline and inference runner optimized for Radxa CM5 (Rockchip RK3588) hardware utilizing **Media Process Platform (MPP)**, **2D Graphics Acceleration (RGA)**, and the **RKNN NPU v2** (6 TOPS).

---

## ⚡ Key Optimizations

- **MPP Decoder**: Decodes H.264 video streams directly into `dma_buf` hardware memory blocks.
- **RGA Hardware Crop & Resize**: Blits and resizes candidate bounding boxes into `(224, 224)` tensor buffers with zero CPU overhead.
- **RKNN NPU Multi-Core Offload**: Spreads inference across NPU Cores 0, 1, and 2 for parallel execution.

---

## 🏗️ C++ Source Architecture

- [`include/rk3588_pipeline.h`](include/rk3588_pipeline.h): Multi-threaded pipeline orchestrator interface.
- [`include/mpp_decoder.h`](include/mpp_decoder.h): Rockchip MPP video decoder wrapper.
- [`include/rga_resizer.h`](include/rga_resizer.h): Rockchip RGA 2D graphics hardware engine wrapper.
- [`include/rknn_engine.h`](include/rknn_engine.h): RKNN NPU execution wrapper.
- [`include/bcs_classifier.h`](include/bcs_classifier.h): 3-Layer MLP BCS classifier head.
- [`src/main.cpp`](src/main.cpp): Executable entrypoint.

---

## 🚀 Building & Running

### 1. Build C++ Target
```bash
chmod +x scripts/build_rk3588.sh
./scripts/build_rk3588.sh
./build/rk3588_cow_bcs sample_cow_video.mp4 models/yolov8n_seg.rknn models/dinov2_vits14.rknn
```

### 2. Python Inference Runner
```bash
python3 scripts/run_rk3588.py --video sample_cow_video.mp4
```
