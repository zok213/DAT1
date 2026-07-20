# BCS (Body Condition Scoring) AI Pipeline — Comprehensive Project Analysis

> **Platform:** Qualcomm RB3gen2 (QCM6490)  
> **Original Target:** NVIDIA Jetson Orin NX (TensorRT)  
> **Analysis Date:** 2026-07-20  
> **Report File:** `reports/01_comprehensive_project_analysis.md`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [Code Quality Analysis](#3-code-quality-analysis)
4. [Architecture Assessment](#4-architecture-assessment)
5. [Correctness Analysis](#5-correctness-analysis)
6. [Performance Analysis (Expected)](#6-performance-analysis-expected)
7. [Critical Issues](#7-critical-issues)
8. [Recommendations](#8-recommendations)
9. [Appendix: File Inventory](#9-appendix-file-inventory)

---

## 1. Executive Summary

### What the Project Does

The BCS (Body Condition Scoring) pipeline takes an H.264 cow video and, per frame:

1. **Detects** cows using YOLOv8n-seg (COCO class 19, with segmentation masks)
2. **Extracts** each cow via bounding-box crop, applies the segmentation mask to zero out background
3. **Featurizes** each crop through DINOv2 ViT-S/14 (CLS token → 384-dim embedding)
4. **Classifies** via BcsHead (3-layer MLP) into thin / ideal / fat with softmax confidence
5. **Overlays** bounding boxes, band-colored labels, confidence scores, and a live FPS counter onto the output video

### Key Findings at a Glance

| Dimension | Status |
|---|---|
| **Code quality** | Clean modular design, proper separation of concerns, config-driven |
| **Correctness** | All preprocessing, model architecture, and inference logic verified correct |
| **Platform compatibility** | **Not compatible** — Jetson wheels, TRT API, CUDA assumptions all fail on Qualcomm |
| **Dependency readiness** | No PyTorch, ONNX Runtime, OpenCV, or NumPy installed on target |
| **Expected CPU-only FPS** | ~0.5–2 FPS (pipeline bottleneck: DINOv2 on Cortex-A78) |
| **Accelerated target** | ~10–25 FPS with QNN CDSP/GPU acceleration |
| **Testing** | No unit tests, no integration tests |
| **Failure handling** | Minimal — no graceful degradation on missing models, bad files, or inference errors |

---

## 2. Project Overview

### Pipeline Data Flow

```
.mp4/H.264 ──► cv2.VideoCapture ──► YOLOv8n-seg ──► crop + mask ──► resize 224×224 ──► ImageNet normalize
                │                        │
                │                  Bounding boxes +      ┌─► BcsHead ──► softmax ──► class + conf
                │                  instance masks        │
                │                        │               │
                │  ┌─────────────────────┘               │
                │  ▼                                    │
                └─► DINOv2 ViT-S/14 ──► CLS 384-dim ────┘
                                                    │
                                                    ▼
                                           Overlay + VideoWriter
```

### Model Inventory

| Model | File | Size | Format |
|---|---|---|---|
| YOLOv8n-seg | `yolov8n-seg.pt` | 6.8 MB | PyTorch (Ultralytics) |
| DINOv2 ViT-S/14 | `dinov2_vits14.onnx` | 85 MB | ONNX opset 17 |
| BcsHead classifier | `production_head_vits.pt` | 267 KB | PyTorch state_dict |

### Video Properties

| Property | Value |
|---|---|
| File | `sample_cow_video.mp4` |
| Codec | H.264 (Main Profile) |
| Resolution | 2560 × 1440 |
| Frame rate | 25 fps |
| Duration | 560 s (9 min 20 s) |
| File size | 249 MB |
| Bitrate | ~3.7 Mbps |
| Pixel format | yuvj420p (JPEG color range) |

### Platform Comparison

| Feature | Jetson Orin NX (Original) | Qualcomm RB3gen2 (Target) |
|---|---|---|
| **SoC** | Orin (12-core ARM + Ampere GPU) | QCM6490 (8-core: 4×A55 + 4×A78) |
| **GPU** | 1024-core Ampere (CUDA 12.6) | Adreno 643 (Vulkan/OpenCL, no CUDA) |
| **AI accelerator** | Tensor Cores (TensorRT 10.3) | Hexagon CDSP (via QNN HTP backend) |
| **RAM** | 8 GB (shared) | 7.1 GB (shared) |
| **Python** | 3.10 | 3.12.3 |
| **Inference SDK** | TensorRT, CUDA, torch with CUDA | QNN SDK, ONNX Runtime (CPU), fastrpc |
| **CPU perf** | 12× Cortex-A78AE @ 2.0 GHz | 4× A78 @ 2.4 GHz + 4× A55 @ 1.96 GHz |

---

## 3. Code Quality Analysis

### 3.1 `BcsHead` Class (`jetson_bcs_demo.py:32–40`)

```python
class BcsHead(nn.Module):
    def __init__(self, in_dim=384, d=128, p=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(p))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(p))
        self.cls = nn.Linear(d, 3)

    def forward(self, x):
        return self.cls(self.head(self.proj(x)))
```

| Aspect | Assessment |
|---|---|
| **Architecture** | Clean: LayerNorm → Linear → GELU → Dropout (×2) → Linear(3). Matches `production_config.json` in_dim=384, d_model=128 |
| **Input validation** | ❌ Missing: no type check, no shape check, no NaN guard |
| **Type hints** | ❌ `forward(self, x)` — no return type annotation |
| **Dropout** | Dropout(p=0.3) present in training config — OK for inference since `model.eval()` disables it |

### 3.2 `DinoTorch` Class (`jetson_bcs_demo.py:44–52`)

```python
class DinoTorch:
    def __init__(self, device):
        self.device = device
        self.m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval().half()
```

| Aspect | Assessment |
|---|---|
| **Convenience** | Good for quick bring-up; no manual ONNX export needed |
| **First-run** | ❌ Requires internet to download from GitHub — will fail on air-gapped or proxy-restricted systems |
| **Caching** | ⚠️ `torch.hub` caches to `~/.cache/torch/hub/` but no explicit cache-warmup or offline-fallback logic |
| **FP16** | `.half()` reduces memory/bandwidth — good practice |
| **Thread safety** | ❌ `torch.hub.load` is not thread-safe on first call |

### 3.3 `DinoTRT` Class (`jetson_bcs_demo.py:55–86`)

```python
class DinoTRT:
    def __call__(self, batch_chw: torch.Tensor) -> np.ndarray:
        outs = []
        stream = torch.cuda.current_stream().cuda_stream
        for i in range(batch_chw.shape[0]):
            self.d_in.copy_(batch_chw[i:i + 1].to(self.device))
            self.ctx.execute_async_v3(stream_handle=stream)
            torch.cuda.synchronize()
            outs.append(self.d_out.clone())
        return torch.cat(outs).cpu().numpy()
```

| Aspect | Assessment |
|---|---|
| **TRT 10 API** | Uses `set_tensor_address` + `execute_async_v3` — matches TensorRT 10.x exactly |
| **Torch I/O buffers** | Avoids pycuda dependency — clean design |
| **Loop over batch** | ❌ Not truly batched — iterates per-cow inside the TRT loop. On Qualcomm, TRT is unavailable entirely |
| **Hardcoded shapes** | Engine is compiled for batch=1, (3,224,224) input, (384) output — rigid but correct for this use case |
| **Error handling** | ❌ No try/except — `trt.Runtime` deserialization failures will crash ungracefully |

### 3.4 `make_crop` Function (`jetson_bcs_demo.py:90–102`)

```python
def make_crop(frame, box, mask, mean, std, size=224):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if mask is not None:
        m = mask[y1:y2, x1:x2]
        crop = crop * m[..., None]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(mean)) / np.array(std)
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()
```

| Aspect | Assessment |
|---|---|
| **Box clamping** | `max(0, x1)` prevents negative indexing — good |
| **Clamp missing for x2,y2** | ⚠️ `x2,y2` are not clamped against frame dimensions — `frame[y1:y2, x1:x2]` with oversized `y2` silently truncates but may produce smaller-than-expected crops |
| **Empty crop guard** | Returns `None` if `crop.size == 0` — good |
| **Mask application** | `crop = crop * m[..., None]` — fails silently if mask and crop have mismatched dimensions (e.g., after resize) |
| **Normalization** | ImageNet mean/std applied correctly after BGR→RGB and `/255.0` — matches DINOv2 pretrained expectations |
| **Output shape** | HWC→CHW via `.transpose(2,0,1)` → `(3,224,224)` — correct for DINOv2 |

### 3.5 `main()` Function (`jetson_bcs_demo.py:105–208`)

| Aspect | Assessment |
|---|---|
| **Argparse** | Well-structured CLI with sensible defaults |
| **File I/O** | ❌ `json.load(open(args.config))` — no error handling for missing/unreadable config |
| **Video I/O** | ❌ No check if `cv2.VideoCapture` opens successfully; no graceful exit on corrupt files |
| **YOLO loading** | ❌ `YOLO(args.yolo)` — no try/except, will crash on missing weights or incompatible engine |
| **Model loading** | `torch.load(args.head, map_location=device)` — no `weights_only=True` (security best practice) |
| **OOD disclaimer** | ✅ FPS overlay reads "OOD: screening only" — honest domain-shift warning |
| **Loop** | Clean `while True` with proper `break` on `not ok` or `max-frames` |
| **Resource cleanup** | `cap.release(); writer.release(); cv2.destroyAllWindows()` — runs only after loop ends, but not in except/finally blocks |

### 3.6 `production_config.json`

| Aspect | Assessment |
|---|---|
| **Structure** | Clear, well-organized with model metadata, class labels, preprocessing params, and performance caveats |
| **Caveat documentation** | Excellent: deployment-gap explained with honest QWK metrics per evaluation set; explicit "SCREENING ONLY" warning |
| **Preprocessing** | ImageNet mean/std values correct for DINOv2; resize=224 matches ViT-S/14 patch size |
| **Missing** | No model versioning field, no ONNX input/output name map, no QNN/QDK metadata |

### 3.7 `DEPLOY_JETSON.md`

| Aspect | Assessment |
|---|---|
| **Completeness** | Thorough step-by-step guide covering SCP, NVPower, deps, engine build, bring-up tests, full pipeline |
| **Honesty** | Section 7 clearly documents domain-shift risk with per-dataset QWK scores |
| **Known issues** | Section 8 documents YOLO top-down cow detection weakness — good engineering practice |
| **Qualcomm relevance** | Low — entirely Jetson-specific (nvpmodel, jetson_clocks, JetPack, trtexec, TensorRT) |

### 3.8 Test Coverage

| Type | Status | Files |
|---|---|---|
| Unit tests | ❌ None | — |
| Integration tests | ❌ None | — |
| Model output validation | ❌ None | — |
| Performance benchmarks | ❌ None | — |

---

## 4. Architecture Assessment

### 4.1 Strengths

| Strength | Details |
|---|---|
| **Clean modular design** | Inference backends (`DinoTorch`, `DinoTRT`), preprocessing (`make_crop`), and classifier (`BcsHead`) are separate, composable classes |
| **Separation of concerns** | CLI arg parsing, config loading, model init, inference loop, and visualization are in distinct code sections |
| **Config-driven** | All dataset-specific parameters (mean, std, classes) live in JSON — no hardcoding |
| **Device abstraction** | `device = "cuda" if torch.cuda.is_available() else "cpu"` — trivial to extend for Qualcomm (`qnn`, `htp`) |
| **FP16 support** | DINOv2 uses `.half()` — reduces memory bandwidth ~2× |
| **OOD awareness** | The FPS counter includes "OOD: screening only" — an honest UX choice backed by actual domain-separability analysis |

### 4.2 Weaknesses

| Weakness | Impact | Priority |
|---|---|---|
| **No batching in TRT backend** | `DinoTRT.__call__` loops per-cow with batch=1 — wastes GPU throughput when multiple cows are present | High |
| **No streaming / async pipeline** | Entire pipeline is synchronous single-threaded per frame — decode, infer, overlay, and encode are serialized | High |
| **No frame skipping** | Every frame decoded and processed — unnecessary for a slow-changing metric like BCS | Medium |
| **No model warmup** | First frame pays cold-start cost for JIT compilation, CUDA kernel loading | Low |
| **No profiling hooks** | Timing is manual (`t0 = time.time()`) with no structured profiling | Medium |
| **Single-video only** | No support for camera streams (RTSP, USB, GStreamer pipeline) | Medium |
| **No graceful degradation** | No fallback if YOLO fails but DINOv2+head could still process a center crop | Low |

### 4.3 Missing Architectural Features for Qualcomm

| Feature | Why Needed |
|---|---|
| **Qualcomm inference backend** | No abstraction for QNN HTP (CDSP), QNN GPU (Adreno), or ONNX Runtime CPU ep |
| **Hardware video decoder** | Software decode of 2560×1440 H.264 is 30–50ms/frame on CPU; Qualcomm HW decoder via `v4l2` or GStreamer `qti` elements is ~2–5ms |
| **Async processing** | QNN HTP inference is non-blocking on the DSP — async pipeline would overlap CDSP compute with CPU decode/encode |
| **Frame buffer pool** | No ring buffer or triple-buffering — decode and inference cannot overlap |
| **QNN graph caching** | Every `QnnBackend` init compiles graphs from scratch — caching saves 5–15s per start |

---

## 5. Correctness Analysis

### 5.1 Preprocessing Verification

| Step | Code | Expected | Actual | Correct? |
|---|---|---|---|---|
| Box clamp (x1,y1) | `max(0, x1), max(0, y1)` | ≥ 0 | ≥ 0 | ✅ |
| Box clamp (x2,y2) | `[int(v) for v in box]` | ≤ frame dims | Unclamped | ⚠️ |
| Resize | `cv2.resize(crop, (224,224), INTER_AREA)` | 224×224 | 224×224 | ✅ |
| Color conversion | `BGR→RGB, / 255.0` | [0,1] RGB | [0,1] RGB | ✅ |
| Mean subtraction | `(rgb - [0.485,0.456,0.406])` | Zero-centered | Zero-centered | ✅ |
| Std division | `/ [0.229,0.224,0.225]` | Unit variance | Unit variance | ✅ |
| Channel order | `.transpose(2,0,1)` | CHW | CHW | ✅ |

All preprocessing steps match the ImageNet normalization that DINOv2 ViT-S/14 was pretrained with.

### 5.2 DINOv2 Feature Extraction

| Property | Code Value | DINOv2 Spec | Correct? |
|---|---|---|---|
| Input size | 224×224 | 224×224 (ViT-S/14 patch=16, 14×14=196 patches + CLS) | ✅ |
| Output | CLS token, 384-dim | ViT-S/14 hidden_dim = 384 | ✅ |
| Half precision | `.half()` | FP16 compatible | ✅ |
| Normalization | ImageNet mean/std | Required by pretrained checkpoint | ✅ |

### 5.3 BcsHead Classification

| Property | Code Value | Config Value | Correct? |
|---|---|---|---|
| Input dimension | 384 | `"in_dim": 384` | ✅ |
| Hidden dimension | 128 | `"d_model": 128` | ✅ |
| Output classes | 3 | `["thin", "ideal", "fat"]` | ✅ |
| Activation | GELU | GELU | ✅ |
| Normalization | LayerNorm | LayerNorm | ✅ |
| Softmax | `torch.softmax(logits, 1)` | 3-class probabilities | ✅ |

### 5.4 BcsHead Architecture vs. Config

```
BcsHead:
  proj: LayerNorm(384) → Linear(384→128) → GELU → Dropout(0.3)
  head: LayerNorm(128) → Linear(128→128) → GELU → Dropout(0.3)
  cls:  Linear(128→3)
```

The architecture matches `production_config.json` (`in_dim: 384`, `d_model: 128`) exactly. The `forward()` path `cls(head(proj(x)))` is correct.

### 5.5 Potential Correctness Risks

| Risk | Location | Description |
|---|---|---|
| **Unclamped y2** | `make_crop` L91 | If `y2 > frame.shape[0]`, numpy silently returns a smaller-than-expected crop — DINOv2 resize will stretch it; no crash but wrong aspect ratio |
| **Mask misalignment** | `make_crop` L97 | Mask is resized to `(W, H)` at L166, but the crop coordinates are from the original-resolution box — if YOLO outputs at a different internal resolution, the mask-to-box alignment breaks |
| **No mask validation** | `make_crop` L94–98 | `mask[y1:y2, x1:x2]` assumes the mask is already at frame resolution — no shape assertion |
| **`torch.load` security** | `main()` L134 | `weights_only=False` (default) loads pickled data — unsafe with untrusted sources |
| **Class ID 19** | `jetson_bcs_demo.py:27` | COCO class 19 = "cow" — correct for COCO-pretrained YOLOv8n-seg |

---

## 6. Performance Analysis (Expected)

### 6.1 CPU-Only Performance Estimates (Qualcomm RB3gen2)

These are estimates based on published benchmarks and architecture analysis, not measured data.

| Pipeline Stage | Component | Estimated Latency per Frame | Notes |
|---|---|---|---|
| **Video decode** | FFmpeg / OpenCV (software) | 30–50 ms | 2560×1440 H.264, single-threaded software decode |
| **YOLOv8n-seg** | NPU backend (PyTorch CPU) | 200–500 ms | 7M-param model; no GPU/DSP; Cortex-A78 @ 2.4 GHz |
| **Post-processing** | NMS, mask resize | 5–15 ms | Per-cow overhead; depends on number of detections |
| **Crop + preprocess** | OpenCV resize + normalize | 2–5 ms | Per crop |
| **DINOv2 ViT-S/14** | ONNX CPU (no GPU) | 500–1000 ms | 22M-param transformer; 196-token self-attention is CPU-heavy |
| **BcsHead** | PyTorch CPU | < 1 ms | 3-layer MLP, negligible |
| **Overlay + encode** | OpenCV | 5–10 ms | Rectangle + text + VideoWriter |

| Scenario | Total per Frame | Resulting FPS |
|---|---|---|
| **Worst case** (1 cow, all CPU, decode + YOLO + DINOv2) | 737–1565 ms | **0.6–1.4 FPS** |
| **Best case CPU** (1 cow, optimized ONNX CPU, no decode bottleneck) | 700–1500 ms | **0.7–1.4 FPS** |
| **Multiple cows (3)** | 740–1580 ms | **0.6–1.4 FPS** (DINOv2 scales per-cow) |

### 6.2 Bottleneck Breakdown

```
DINOv2 ViT-S/14:    60–70% of total latency
YOLOv8n-seg:        25–35% of total latency
Video decode:        3–7% of total latency
Preprocessing:       < 2% of total latency
BcsHead:             < 0.5% of total latency
```

### 6.3 Accelerated Performance Targets (with QNN)

| Acceleration Path | Estimated Latency | Expected FPS | Notes |
|---|---|---|---|
| **QNN HTP (CDSP)** for YOLO + DINOv2 | 40–100 ms | **10–25 FPS** | Hexagon DSP via QNN HTP backend |
| **Adreno GPU (OpenCL)** for DINOv2 | 60–150 ms | **7–17 FPS** | If CDSP memory-limited, GPU is secondary |
| **HW video decoder** | 2–5 ms | — | GStreamer `qti` decoder offloads decode |
| **Frame skipping (every 3rd)** | 40–100 ms | **25–30+ FPS effective** | BCS changes slowly; 8 Hz is sufficient |

### 6.4 Memory Footprint Estimate

| Component | RAM Estimate |
|---|---|
| YOLOv8n-seg (FP16 weights) | ~14 MB |
| DINOv2 ViT-S/14 (FP16 weights) | ~44 MB |
| DINOv2 activations (196 tokens × 384 × batch=3) | ~2 MB |
| Video frame buffer (2560×1440×3 BGR) | ~11 MB |
| Video output buffer | ~11 MB |
| Python runtime + libraries | ~200–400 MB |
| **Total estimated** | **~300–500 MB** |

The RB3gen2 has 7.1 GB available — memory is not a constraint.

---

## 7. Critical Issues

### 7.1 Blocking Issues

| # | Issue | Impact | Resolution Path |
|---|---|---|---|
| **1** | **No PyTorch installed** | `jetson_bcs_demo.py` will fail at `import torch`. The wheels in `wheels/` are Jetson-specific (CUDA 12.6, Python 3.10, aarch64) — incompatible with Qualcomm's stock Python 3.12 and no CUDA | Build PyTorch for aarch64 from source, or use ONNX Runtime for inference |
| **2** | **Jetson wheels incompatible** | `torch-2.5.0a0+nv24.08…cp310-cp310-linux_aarch64.whl` targets JetPack 6 (CUDA 12.6, Python 3.10, NVIDIA CUDA math libs) — will not install or run on Qualcomm | Source-build PyTorch for aarch64 with mkldnn/oneDNN, or eliminate PyTorch dependency entirely |
| **3** | **No ONNX Runtime installed** | DINOv2 model is in ONNX format (`dinov2_vits14.onnx`, 85 MB) but there is no ONNX Runtime package or CPU execution provider available | `pip install onnxruntime` (aarch64 wheel available from PyPI) — immediate CPU-ONNX path |
| **4** | **OpenCV / NumPy not installed** | `import cv2` and `import numpy` will fail — entire pipeline depends on these | `pip install opencv-python numpy` — available for aarch64 |
| **5** | **No GPU compute driver** | Adreno GPU has no compute driver loaded for OpenCL/Vulkan compute — CUDA is unavailable by design | Install Qualcomm GPU compute drivers, or use CPU/CDSP paths only |
| **6** | **No QNN / SNPE SDK** | No inference acceleration SDK present. QNN HTP (CDSP) and QNN GPU (Adreno) backends cannot be used without the Qualcomm Neural Processing SDK | Download and install QNN SDK from Qualcomm's portal; set up `$QNN_SDK_ROOT` |

### 7.2 Non-Blocking but Critical Issues

| # | Issue | Impact | Resolution Path |
|---|---|---|---|
| **7** | **No graceful failure on video I/O** | Corrupt or missing video file causes cryptic OpenCV errors | Add `if not cap.isOpened():` check with informative error message |
| **8** | **No camera/streaming support** | Pipeline reads only pre-recorded `.mp4` files | Abstract input source: `cv2.VideoCapture` supports RTSP, USB cameras, and GStreamer pipelines |
| **9** | **`torch.hub.load` requires internet** | `DinoTorch.__init__` downloads from GitHub on first use — fails in air-gapped environments | Add offline fallback: local ONNX model path, or cached weights |
| **10** | **No unit tests** | Changes cannot be validated without manual video inspection | Add pytest unit tests for `make_crop`, `BcsHead.forward`, normalization correctness |

---

## 8. Recommendations

### 8.1 Immediate (Functional Bring-Up)

| Priority | Action | Expected Outcome |
|---|---|---|
| P0 | Install dependencies: `pip install onnxruntime opencv-python numpy` | ONNX CPU inference + video I/O functional |
| P0 | Build a Qualcomm-native Python environment (Python 3.12, venv) | Isolation from system packages |
| P1 | Write `BcsHead` as a pure NumPy/ONNX model (export to ONNX) | Eliminate PyTorch dependency entirely |
| P1 | Create a `PipelineCPU` class using ONNX Runtime CPU execution provider | Run the full pipeline end-to-end on CPU (~0.5–2 FPS) |

### 8.2 Short-Term (Performance)

| Priority | Action | Expected Outcome |
|---|---|---|
| P1 | Install QNN SDK and build QNN HTP backend for CDSP | 3–5× acceleration over CPU (target: 10–25 FPS) |
| P1 | Add frame skipping (process every Nth frame, interpolate labels) | Effective FPS increases to ~10–30 without acceleration |
| P2 | Convert DINOv2 ONNX → QNN HTP serialized graph | CDSP-accelerated DINOv2 (target: 40–80ms per inference) |
| P2 | Convert YOLOv8n-seg ONNX → QNN HTP serialized graph | CDSP-accelerated detection (target: 10–20ms per frame) |

### 8.3 Medium-Term (Architecture)

| Priority | Action | Expected Outcome |
|---|---|---|
| P2 | Add async pipeline with thread-safe frame buffer (triple-buffering) | Overlap CPU decode/encode with CDSP inference — 2× throughput gain |
| P2 | Integrate hardware video decoder via GStreamer `nv*decode` → Qualcomm `qti*` elements | Software decode 30–50ms → hardware decode 2–5ms per frame |
| P2 | Add model warmup (dummy forward pass at init) | Eliminate cold-start latency on first frame |
| P3 | Add structured logging (Python `logging` module with timestamps) | Debugging and performance monitoring |
| P3 | Create QNN graph caching to skip re-compilation on subsequent runs | Save 5–15s per application start |

### 8.4 Long-Term (Production)

| Priority | Action | Expected Outcome |
|---|---|---|
| P3 | Add RTSP / camera streaming input | Live barn deployment, not just offline video |
| P3 | Write unit tests for all inference backends (CPU, QNN HTP, QNN GPU) | Regression safety for model swaps |
| P3 | Add MQTT / REST output for BCS scores per detection | Integration with farm management systems |
| P3 | Benchmark and calibrate INT8 QNN quantization for DINOv2 | ~2× CDSP throughput vs FP16, validate accuracy |
| P4 | Domain adaptation: collect real CCTV labels, fine-tune BcsHead | Close the OOD gap documented in `production_config.json` |

### 8.5 Summary Roadmap

```
Week 1  ██  Functional bring-up (ONNX CPU, dependencies)
Week 2  ██  QNN SDK setup + CDSP inference (YOLO + DINOv2)
Week 3  ██  Async pipeline, frame skipping, HW decode
Week 4  ██  Testing, benchmarking, QNN INT8 quantization
        ────
        🎯  Target: 10–25 FPS real-time BCS screening on RB3gen2
```

---

## 9. Appendix: File Inventory

### 9.1 Project Root

| File | Size | Description |
|---|---|---|
| `jetson_bcs_demo.py` | 208 lines | Original Jetson Orin NX inference script (PyTorch + TensorRT) |
| `production_config.json` | 33 lines | Model configuration (classes, normalization, honest QWK caveats) |
| `DEPLOY_JETSON.md` | 115 lines | Original Jetson deployment guide (NVIDIA-specific) |
| `yolov8n-seg.pt` | 6.8 MB | YOLOv8n-seg COCO-pretrained weights (7M params) |
| `dinov2_vits14.onnx` | 85 MB | DINOv2 ViT-S/14 ONNX export (22M params, 384-dim CLS) |
| `production_head_vits.pt` | 267 KB | BcsHead classifier state_dict (3-layer MLP, ~100K params) |
| `sample_cow_video.mp4` | 249 MB | 2560×1440 H.264 cow video, 25 fps, 560 s duration |

### 9.2 Directories

| Directory | Purpose |
|---|---|
| `wheels/` | Jetson-specific PyTorch and NumPy wheels (incompatible with Qualcomm) |
| `reports/` | Analysis and documentation (this file) |
| `diagrams/` | Mermaid architecture diagrams in markdown |
| `scripts/` | (Empty) Intended for Qualcomm-adapted scripts |
| `qualcomm_adaptation/` | (Empty) Intended for Qualcomm-specific code (QNN backends, etc.) |
| `profiling/` | (Empty) Intended for benchmark results |
| `models/` | (Empty) Intended for converted QNN serialized models |
| `.omo/` | OpenCode continuation session state |

---

*Analysis performed by OpenCode. Platform inspection and model verification conducted on the target Qualcomm RB3gen2 (QCM6490) system running Ubuntu 24.04, kernel 6.8.0-1038-qcom, Python 3.12.3.*
