# 🐄 Qualcomm RB3 Gen2 (QCM6490) Deployment

This directory contains the production-grade Python and C++ pipeline optimized for **Qualcomm RB3 Gen2 (QCM6490)** leveraging the **Hexagon CDSP / HTP Backend**, **Qualcomm AI Runtime (QAIRT)**, and **ION DMA-BUF Zero-Copy Memory**.

---

## ⚡ Key Optimizations

- **QNN HTP Acceleration**: Offloads matrix multiplication to the Hexagon DSP (12–15 INT8 TOPS) at sub-3W power draw.
- **ION DMA-BUF Zero-Copy**: Uses Linux DMA-BUF file descriptors (`/dev/dma_heap/system`) to pass frames directly from V4L2 HW video decoder to QNN HTP without CPU memory copies.
- **GStreamer V4L2 HW Decoding**: Uses `msm_vidc_decoder` hardware block for low-power 1080p decoding.

---

## 🏗️ Python & QNN Architecture

- [`config.py`](config.py): Configuration dataclass (`BCSConfig`) for runtime parameters and preprocessing.
- [`pipeline.py`](pipeline.py): End-to-end pipeline handling GStreamer V4L2 decode, YOLO detection, DINOv2 feature extraction, and `BcsHead` scoring.
- [`qnn_backend.py`](qnn_backend.py): Integration with Qualcomm AI Runtime (QAIRT) Python API (`DinoQNN`).
- [`__main__.py`](__main__.py): Execution entrypoint and CLI suite.
- [`Dockerfile.qualcomm`](Dockerfile.qualcomm): Container definition for Qualcomm Ubuntu environment.

---

## 🚀 Running Inference

```bash
# Run Qualcomm QNN pipeline with V4L2 Hardware Decode
python3 -m qualcomm_adaptation --yolo models/yolov8n_seg.onnx --dino qnn_models/dinov2_fp32_net.json --video sample_cow_video.mp4 --hw-decode
```
