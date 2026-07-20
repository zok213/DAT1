# BCS Cow AI Pipeline — Optimization Roadmap for Qualcomm RB3gen2

> **Target Platform:** Qualcomm RB3gen2 (8-core: 4×Cortex-A55 + 4×Cortex-A78, 7.1 GB RAM, Adreno 618 GPU, Hexagon CDSP via fastrpc)
> **Current Baseline:** ~1–3 FPS (CPU-only PyTorch FP32 at 2560×1440)
> **Goal:** 25+ FPS real-time inference at acceptable accuracy

---

## Table of Contents

1. [Pipeline Baseline Analysis](#1-pipeline-baseline-analysis)
2. [Optimization Hierarchy by Impact](#2-optimization-hierarchy-by-impact)
3. [Detailed Optimization Descriptions (Top 10)](#3-detailed-optimization-descriptions-top-10)
4. [Combined Optimization Strategy](#4-combined-optimization-strategy)
5. [Accuracy Impact Analysis](#5-accuracy-impact-analysis)
6. [Implementation Priority Matrix](#6-implementation-priority-matrix)
7. [Risk Assessment](#7-risk-assessment)
8. [Model Conversion & Deployment Workflow](#8-model-conversion--deployment-workflow)
9. [Benchmarking Methodology](#9-benchmarking-methodology)
10. [Monitoring & Maintenance](#10-monitoring--maintenance)

---

## 1. Pipeline Baseline Analysis

### 1.1 Current Pipeline Stages

| Stage | Model | Compute | Input Size | Output | Est. CPU Time (ms) | % of Pipeline |
|---|---|---|---|---|---|---|
| Video Decode | — | CPU (OpenCV) | 2560×1440@25 | BGR frame | ~25–40 | 3–5% |
| YOLOv8n-seg Inference | YOLOv8n-seg | CPU (PyTorch) | 640×640 (internal) | boxes + masks | ~200–350 | 50–60% |
| Background Masking | — | CPU (OpenCV) | per-crop 224×224 | masked crop | ~5–10 | 1–2% |
| DINOv2 ViT-S/14 | dinov2_vits14 | CPU (PyTorch FP32) | 224×224 → (1,384) | feature vec | ~150–300 | 35–45% |
| BcsHead Classifier | BcsHead | CPU (PyTorch) | 384-d → 3 | softmax scores | ~1–2 | <1% |
| Overlay + Encode | — | CPU (OpenCV) | 2560×1440 | annotated frame | ~5–10 | 1–2% |

### 1.2 Key Bottlenecks

1. **DINOv2 ViT-S/14** — 88 MB model, 86M parameters, attention-heavy. This is the dominant cost when multiple cows are in frame.
2. **YOLOv8n-seg** — ~3.2M parameters + segmentation head. Scales with number of detected objects.
3. **Video Decode** — Software decoding of 2560×1440 H.264 on A55 cores is inefficient.
4. **Background Masking** — Per-cow mask resize + pixel multiplication adds up with many cows.

### 1.3 RB3gen2 Hardware Constraints

| Resource | Available | Constraint |
|---|---|---|
| CPU big (A78) | 4 cores @ 2.4 GHz | Shared with OS + other processes |
| CPU LITTLE (A55) | 4 cores @ 1.8 GHz | Efficient but slow for vectorized compute |
| RAM | 7.1 GB | DINOv2 (88 MB) + YOLO (7 MB) + video buffers — memory pressure at high res |
| Adreno 618 GPU | Yes | No OpenCL/Vulkan compute driver by default — only rendering |
| Hexagon CDSP | Yes | Available via fastrpc — QNN SDK needed for access |
| NPU (HVX) | No | RB3gen2 has CDSP but dedicated tensor accelerator may be absent |

---

## 2. Optimization Hierarchy by Impact

### 2.1 Tier 1: "Low Hanging Fruit" (2–5× improvement, minimal effort)

| # | Optimization | Est. Speedup | Effort | Complexity | Risk | Cumulative FPS |
|---|---|---|---|---|---|---|
| 1 | **Frame skipping** — process every 2nd/3rd frame | 2–3× | 5 min | Trivial | None | 3–9 |
| 2 | **Reduce input resolution** — 1280×720 instead of 2560×1440 | 1.5–2× | 10 min | Trivial | Very Low | 5–12 |
| 3 | **Reduce DINOv2 input size** — 168×168 instead of 224×224 | 1.5–2× | 5 min | Trivial | Low | 7–15 |
| 4 | **YOLO confidence threshold tuning** — raise to 0.5–0.6 | 1.1–1.3× | 5 min | Trivial | Low | 8–16 |
| 5 | **BcsHead → NumPy** — remove PyTorch dependency for classifier | Marginal | 30 min | Easy | None | 8–16 |

**Total Tier 1 impact:** ~5–8× over baseline (5–16 FPS depending on cow count)

### 2.2 Tier 2: "Software Optimization" (3–5× improvement, moderate effort)

| # | Optimization | Est. Speedup | Effort | Complexity | Risk | Cumulative FPS |
|---|---|---|---|---|---|---|
| 6 | **ONNX Runtime CPU with threads** — convert all models to ONNX, run with 4–8 threads | 1.5–2× | 1 hr | Medium | Low | 10–20 |
| 7 | **Multi-threaded decode + infer pipeline** — overlap decode, YOLO, and DINOv2 across threads | 1.3–1.5× | 2 hr | Medium | Medium | 12–25 |
| 8 | **Remove background masking** — skip segmentation mask step entirely | 1.1–1.2× | 10 min | Easy | Low | 13–27 |
| 9 | **Batch all cows per frame** — single DINOv2 call with batch dimension | 1.2–2× | 1 hr | Medium | Low | 15–30 |
| 10 | **Quantize YOLO to FP16/INT8** — via ONNX Runtime dynamic quantization | 1.5–2× | 2 hr | Medium | Medium | 18–35 |

**Total Tier 2 impact:** ~10–15× over baseline (18–35 FPS)

### 2.3 Tier 3: "Hardware Acceleration" (5–10× improvement, high effort)

| # | Optimization | Est. Speedup | Effort | Complexity | Risk | Cumulative FPS |
|---|---|---|---|---|---|---|
| 11 | **QNN SDK → DINOv2 on CDSP** — compile DINOv2 to Hexagon DSP via Qualcomm Neural Networks SDK | 3–5× | 1–2 days | Hard | High | 25–50 |
| 12 | **QNN SDK → YOLO on CDSP** — compile YOLOv8n-seg to Hexagon DSP | 3–5× | 1–2 days | Hard | High | 30–60 |
| 13 | **QNN Adreno GPU backend** — OpenCL compute backend for QNN | 2–3× over CDSP | 1–2 days | Hard | Very High | 40–80 |
| 14 | **INT8 quantization via QNN** — 4× memory reduction + throughput gain | 2× throughput | 1 day | Hard | High | 50–100 |
| 15 | **GStreamer HW video decode** — VAAPI/v4l2 hardware decoder | 5–10× for decode | 1 day | Hard | Medium | 55–110 |

**Total Tier 3 impact:** ~25–40× over baseline (55–110 FPS)

### 2.4 Tier 4: "Architecture-level" (10–20×, very high effort)

| # | Optimization | Est. Speedup | Effort | Complexity | Risk | Cumulative FPS |
|---|---|---|---|---|---|---|
| 16 | **Replace DINOv2 with distilled ViT** — TinyViT-5M or MobileViT-S | 3–5× | 2–3 days | Very Hard | High | 60–100 |
| 17 | **Replace YOLOv8 with YOLOv10-nano** — newer arch, fewer params | 1.5–2× | 1 day | Hard | Medium | 65–110 |
| 18 | **Tensor decomposition / pruning of DINOv2** — SVD + structured pruning | 2–3× | 1–2 weeks | Research | Very High | 70–120 |
| 19 | **Custom lightweight feature extractor** — train a small CNN/ViT from scratch on cow data | 5–10× | 2–4 weeks | Research | Very High | 80–130 |
| 20 | **C++ implementation with Hexagon SDK** — native code on CDSP | 2–3× | 2–4 weeks | Very Hard | High | 90–150 |

**Total Tier 4 impact:** ~40–60× over baseline (90–150 FPS theoretical maximum)

---

## 3. Detailed Optimization Descriptions (Top 10)

### 3.1 Frame Skipping

**Description:** Process only every Nth frame (N=2 or 3) from the video, copy the previous result for skipped frames. Cows move slowly in typical barn footage, so temporal redundancy is high.

**Expected speedup:** 2–3× (skip every other frame = 2×, skip 2 of 3 = 3×)

**Code changes:**
```python
FRAME_SKIP = 2  # process every 2nd frame
frame_count = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    if frame_count % FRAME_SKIP != 0:
        writer.write(last_annotated_frame)  # reuse previous
        frame_count += 1
        continue
    # ... full inference ...
    last_annotated_frame = frame.copy()
    frame_count += 1
```

**Accuracy impact:** Minimal — cows at 25 fps move <1 pixel between frames. BCS changes on timescale of days.

**Risk:** None. Trivially reversible.

---

### 3.2 Reduce Input Resolution

**Description:** Downscale the input frame from 2560×1440 to 1280×720 before feeding through the pipeline. YOLO internal resolution is already 640×640, so the primary benefit is in decode time and memory bandwidth.

**Expected speedup:** 1.5–2× (decode is 4× fewer pixels; DINOv2 crop is unaffected since crops are always resized to 224×224)

**Code changes:**
```python
# After cap.read()
frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
```

**Accuracy impact:** Minor — detection at 1280×720 is generally fine for cows in barns. May miss very distant cows. YOLO's internal 640×640 already means high-res input is downscaled internally anyway.

**Risk:** Very Low. Test with representative footage.

---

### 3.3 Reduce DINOv2 Input Size

**Description:** Resize cow crops to 168×168 instead of 224×224 before DINOv2 feature extraction. DINOv2 ViT-S/14 processes 14×14 patches; at 224×224 that's 16×16 = 256 patches. At 168×168, it's 12×12 = 144 patches — a 44% reduction in sequence length, which corresponds to ~1.7× speedup in the attention layers (O(n²) complexity).

**Expected speedup:** 1.5–2× per DINOv2 call (scales quadratically with patch count)

**Code changes:**
```python
DINO_INPUT_SIZE = 168  # was 224
# In make_crop:
crop = cv2.resize(crop, (DINO_INPUT_SIZE, DINO_INPUT_SIZE), interpolation=cv2.INTER_AREA)
```

**Accuracy impact:** Low-to-medium. DINOv2 was trained on 224×224; 168×168 means fewer patches and coarser feature extraction. BCS may lose fine body condition detail. Needs validation with a held-out test set.

**Risk:** Low. Fast to test — change one constant.

---

### 3.4 YOLO Confidence Threshold Tuning

**Description:** Raise `--conf` from the default 0.25 to 0.5–0.6. This reduces the number of false-positive detections, which directly reduces the number of DINOv2 forward passes (the most expensive stage).

**Expected speedup:** 1.1–1.3× (proportional to reduction in detected cows)

**Code changes:** Single command-line argument change:
```bash
--conf 0.5    # instead of 0.25
```

**Accuracy impact:** Low — false positives are rare for cows in typical barn footage at 0.25. Raising to 0.5 still detects all visible cows. Very unlikely to miss true positives.

**Risk:** None. Quick to validate.

---

### 3.5 BcsHead → NumPy Implementation

**Description:** The BcsHead classifier is 3 linear layers with LayerNorm, GELU, and Dropout. At inference with no training, this can be expressed as pure NumPy operations, eliminating the PyTorch dependency and tensor conversion overhead.

**Expected speedup:** Marginal (~1 ms per frame). Not about speed — removes PyTorch as a dependency for the classification step.

**Code changes:**
```python
import numpy as np

class BcsHeadNumpy:
    def __init__(self, state_dict):
        # Extract weights and biases from PyTorch state_dict
        self.ln1_w, self.ln1_b = state_dict['proj.0.weight'].numpy(), state_dict['proj.0.bias'].numpy()
        self.fc1_w, self.fc1_b = state_dict['proj.1.weight'].numpy(), state_dict['proj.1.bias'].numpy()
        self.ln2_w, self.ln2_b = state_dict['head.0.weight'].numpy(), state_dict['head.0.bias'].numpy()
        self.fc2_w, self.fc2_b = state_dict['head.1.weight'].numpy(), state_dict['head.1.bias'].numpy()
        self.cls_w, self.cls_b = state_dict['cls.weight'].numpy(), state_dict['cls.bias'].numpy()

    def layer_norm(self, x, w, b, eps=1e-5):
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return w * (x - mean) / np.sqrt(var + eps) + b

    def __call__(self, x):
        x = self.layer_norm(x, self.ln1_w, self.ln1_b)
        x = np.maximum(0, x @ self.fc1_w.T + self.fc1_b)  # GELU ≈ ReLU at inference
        x = self.layer_norm(x, self.ln2_w, self.ln2_b)
        x = np.maximum(0, x @ self.fc2_w.T + self.fc2_b)
        logits = x @ self.cls_w.T + self.cls_b
        exp = np.exp(logits - logits.max(-1, keepdims=True))
        return exp / exp.sum(-1, keepdims=True)
```

**Accuracy impact:** None — mathematically identical (linear algebra is exact). GELU → ReLU substitution has negligible effect.

**Risk:** None.

---

### 3.6 ONNX Runtime CPU with Threads

**Description:** Convert YOLOv8n-seg and DINOv2 ViT-S/14 to ONNX format, then run inference via ONNX Runtime with 4–8 CPU threads. ONNX Runtime applies operator fusion, constant folding, and memory layout optimizations that PyTorch eager mode does not.

**Expected speedup:** 1.5–2× over PyTorch CPU

**Code changes:**

_Model conversion:_
```bash
# DINOv2: export from PyTorch to ONNX
python -c "
import torch; import torch.onnx
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'dinov2_vits14.onnx',
    input_names=['input'], output_names=['output'],
    opset_version=17, dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
"

# YOLOv8: ultralytics exports to ONNX natively
yolo export format=onnx imgsz=640
```

_Inference code:_
```python
import onnxruntime as ort
sess = ort.InferenceSession('dinov2_vits14.onnx',
    providers=[('CPUExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'})])
feats = sess.run(None, {'input': batch.numpy()})[0]
```

**Accuracy impact:** Minimal — ONNX uses the same weights. FP32 vs FP32 is bit-exact.

**Risk:** Low. Well-understood path. ONNX Runtime is mature.

---

### 3.7 Multi-threaded Decode + Infer Pipeline

**Description:** Use a producer-consumer pattern with 3 threads:
- **Thread 1 (Decoder):** Reads video frames, downscales, pushes to a queue
- **Thread 2 (YOLO):** Runs detection on frames from the queue
- **Thread 3 (DINOv2 + BcsHead):** Runs feature extraction and classification on detected crops

This overlaps I/O-bound decode with compute-bound inference.

**Expected speedup:** 1.3–1.5× (reduces pipeline stalls from sequential execution)

**Code changes:**
```python
import queue
import threading
import concurrent.futures

frame_queue = queue.Queue(maxsize=2)
crop_queue = queue.Queue(maxsize=4)

def decoder(cap):
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_queue.put(cv2.resize(frame, (1280, 720)))

def yolo_worker(yolo, conf):
    while True:
        frame = frame_queue.get()
        results = yolo.predict(frame, classes=[19], conf=conf, verbose=False)
        # ... extract crops, push to crop_queue ...
        crop_queue.put((frame, boxes, crops))

def dino_worker(dino, head):
    while True:
        frame, boxes, crops = crop_queue.get()
        if crops:
            feats = dino(torch.stack(crops))
            # ... classify, overlay ...
```

**Accuracy impact:** None — same computation, different scheduling.

**Risk:** Medium. Thread safety requires careful management. Python GIL limits true parallelism; consider `multiprocessing` with shared memory for CPU-bound stages.

---

### 3.8 Remove Background Masking

**Description:** Skip the segmentation mask application step. YOLO already provides bounding boxes; the background masking (`crop * mask[..., None]`) adds per-cow resize + multiply overhead with minimal benefit for BCS.

**Expected speedup:** 1.1–1.2× (eliminates mask resize + per-pixel multiplication per cow)

**Code changes:**
```python
# In make_crop — simply ignore the mask parameter:
def make_crop(frame, box, mean, std, size=224):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return None
    # Remove: if mask is not None: m = mask[y1:y2, x1:x2]; crop = crop * m[..., None]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    # ... rest unchanged ...
```

**Accuracy impact:** Low — the background around a tight cow bounding box is mostly floor/wall, which DINOv2 largely ignores due to ImageNet-normalized training. BCS focuses on body contours, not background.

**Risk:** Low. Test with 10–20 frames to verify no significant feature drift.

---

### 3.9 Batch All Cows per Frame for Single DINOv2 Call

**Description:** Stack all cow crops from a single frame into a batch tensor and make a single DINOv2 forward pass instead of N sequential calls. DINOv2 benefits from vectorized batch processing.

**Expected speedup:** 1.2–2× (scales with number of cows; batch of 4–6 cows gives ~1.5–2×)

**Code changes:**
```python
# Current (per-cow loop):
# for crop in crops: feats = dino(crop.unsqueeze(0))

# Optimized (batched):
if crops:
    batch = torch.stack(crops)  # (K, 3, 224, 224)
    feats = dino(batch)         # (K, 384) — single forward pass
```

**Accuracy impact:** None — mathematically identical to per-cow calls.

**Risk:** Low. Implementation already partially batching in the current code (`torch.stack(crops)` in `jetson_bcs_demo.py:172`). Just need to ensure the DINOv2 backend handles batch dimension > 1.

---

### 3.10 Quantize YOLO to FP16/INT8

**Description:** Convert YOLOv8n-seg from FP32 to FP16 or INT8 precision using ONNX Runtime's dynamic quantization or QAT (Quantization-Aware Training). FP16 halves memory and bandwidth; INT8 provides 4× reduction.

**Expected speedup:** 1.5–2× (FP16) or 2–3× (INT8)

**Code changes:**

_ONNX Runtime dynamic quantization:_
```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'yolov8n-seg.onnx'
model_int8 = 'yolov8n-seg.int8.onnx'
quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QInt8)
```

_QNN INT8 quantization (more aggressive):_ Requires calibration dataset:
```bash
qnn-quantize --model yolov8n-seg.onnx \
    --calibration-dataset ./calib_data/ \
    --activation-bitwidth 8 --weight-bitwidth 8 \
    --output-path yolov8n-seg.qnn
```

**Accuracy impact:** Medium. INT8 quantization can cause 1–3% mAP drop for detection models. For BCS (classification after cropping), the impact is likely smaller — the bounding box just needs to be "good enough." FP16 is essentially lossless.

**Risk:** Medium. INT8 requires calibration data and per-tensor validation. Class imbalance in the calibration set can cause biased quantization.

---

## 4. Combined Optimization Strategy

### 4.1 Phase 1: Quick Wins (Day 1 — ~4 hours)

**Goal:** 5–10 FPS with minimal code changes

| Step | Action | Time | Est. FPS |
|---|---|---|---|
| 1 | Apply frame skipping (every 2nd frame) | 5 min | 3–6 |
| 2 | Reduce input to 1280×720 | 10 min | 5–10 |
| 3 | Set `--conf 0.5` | 5 min | 5–10 |
| 4 | Remove background masking | 10 min | 6–12 |
| 5 | Reduce DINOv2 input to 168×168 | 5 min | 8–15 |
| 6 | Validate accuracy on 100-frame test set | 30 min | — |
| 7 | BcsHead → NumPy conversion | 30 min | 8–15 |

**Deliverable:** `bcs_optimized_v1.py` — modified inference script hitting ~8–15 FPS

**Verification:**
```bash
python bcs_optimized_v1.py --video sample.mp4 --conf 0.5 --skip 2 --resize 1280x720 --dino-size 168
```

---

### 4.2 Phase 2: Software Optimization (Week 1 — ~2 days)

**Goal:** 10–20 FPS with moderate engineering effort

| Step | Action | Time | Est. FPS |
|---|---|---|---|
| 1 | Convert YOLOv8n-seg to ONNX | 30 min | — |
| 2 | Convert DINOv2 ViT-S/14 to ONNX | 30 min | — |
| 3 | ONNX Runtime CPU with 6 threads | 1 hr | 10–18 |
| 4 | Multi-threaded pipeline (3 threads) | 2 hr | 12–22 |
| 5 | Batch DINOv2 per frame (already partially done) | 30 min | 12–22 |
| 6 | YOLO INT8 quantization via ONNX RT | 1 hr | 15–25 |
| 7 | Full regression test + accuracy validation | 2 hr | — |

**Deliverable:** `bcs_optimized_v2.py` — ONNX Runtime threading pipeline hitting ~15–25 FPS

**Architecture at this stage:**
```
[Decoder Thread] → queue → [YOLO Thread] → queue → [DINOv2+BcsHead Thread] → output
     (CPU)                        (ONNX CPU 6T)               (ONNX CPU + NumPy)
```

---

### 4.3 Phase 3: Hardware Acceleration (Week 2 — 3–5 days)

**Goal:** 15–30 FPS with Qualcomm QNN SDK

| Step | Action | Time | Est. FPS |
|---|---|---|---|
| 1 | Install Qualcomm QNN SDK (Snapdragon Neural Processing Engine) | 1 day | — |
| 2 | Convert DINOv2 to QNN and deploy on CDSP | 1–2 days | 20–35 |
| 3 | Convert YOLOv8n-seg to QNN and deploy on CDSP | 1–2 days | 25–40 |
| 4 | QNN INT8 quantization with calibration | 1 day | 30–50 |
| 5 | Investigate Adreno GPU OpenCL compute path | 1–2 days | 40–60 |
| 6 | GStreamer hardware video decode | 1 day | 45–65 |

**Deliverable:** `bcs_qnn.py` — QNN-accelerated pipeline hitting ~25–60 FPS

**Architecture at this stage:**
```
[Decoder: HW] → queue → [YOLO: CDSP] → queue → [DINOv2: CDSP/GPU] → [BcsHead: CPU] → output
 (GStreamer v4l2)       (QNN HTP backend)         (QNN HTP/GPU backend)    (NumPy)
```

---

### 4.4 Phase 4: Architecture Optimization (Month 1+ — 2–4 weeks)

**Goal:** 25+ FPS (sustained) with model substitution

| Step | Action | Time | Est. FPS |
|---|---|---|---|
| 1 | Train/evaluate distilled DINOv2 (TinyViT-5M or MobileViT-S) | 1–2 weeks | 30–50 |
| 2 | Replace YOLOv8 with YOLOv10-nano | 2–3 days | 35–55 |
| 3 | Retrain BcsHead for new feature extractor | 3–5 days | — |
| 4 | Tensor decomposition of DINOv2 (SVD on attention projections) | 1–2 weeks | 40–60 |
| 5 | Full end-to-end optimization + stability testing | 1 week | 40–60 |

**Deliverable:** `bcs_production.py` — production-ready optimized pipeline

---

### 4.5 Optimization Decision Tree

```
START: 1–3 FPS CPU-only
│
├── TIER 1 (Day 1): Frame skip + resize + conf tuning
│   └── 8–15 FPS achieved?
│       ├── YES → Continue to Tier 2
│       └── NO  → Check bottlenecks:
│                   ├── Many cows → batch DINOv2
│                   └── Single cow → DINOv2 dominates → resize smaller
│
├── TIER 2 (Week 1): ONNX + threading
│   └── 15–25 FPS achieved?
│       ├── YES → Continue to Tier 3
│       └── NO  → Profile:
│                   ├── CPU bound → QNN CDSP
│                   └── I/O bound → HW decode
│
├── TIER 3 (Week 2): QNN hardware acceleration
│   └── 25–60 FPS achieved?
│       ├── YES → Production deployment
│       └── NO  → Model architecture changes
│
└── TIER 4 (Month 1+): Model distillation
    └── 40–60+ FPS → Production deployment
```

---

## 5. Accuracy Impact Analysis

### 5.1 Impact Summary

| Optimization | Accuracy Impact | Validation Required? | Mitigation |
|---|---|---|---|
| Frame skipping | None (output reuse for skipped frames) | No | Verify FPS counter is frame-rate-aware |
| Resolution 1280×720 | <0.5% BCS QWK drop | Yes — 50-frame test | Compare per-cow scores vs baseline |
| DINOv2 168×168 | 1–3% QWK drop possible | Yes — full test set | Tune BcsHead for 168×168 inputs |
| YOLO conf 0.5 | <0.1% drop (rare false negatives) | Minimal | Verify no cows missed in 100 frames |
| BcsHead → NumPy | None (mathematically identical) | No | Verify output diff < 1e-6 |
| ONNX Runtime CPU | None (FP32) or <0.1% (FP16) | Minimal | Compare outputs vs PyTorch baseline |
| Multi-threading | None (same computation) | No | Verify determinism with same inputs |
| Remove masking | 0.5–1% QWK drop (background noise) | Yes — 100-frame test | Train BcsHead without masking if needed |
| Batch DINOv2 | None (same computation) | No | Verify numerical equivalence |
| YOLO INT8 | 1–2% mAP drop → negligible for BCS | Yes — 200-frame test | Use FP16 for safety-critical deployments |
| QNN CDSP | <0.1% (FP16/INT8 differences) | Yes — full validation | Use FP16 first, INT8 only if accuracy holds |
| Model substitution (TinyViT) | 3–8% QWK drop possible | Yes — full retrain + test | Requires BcsHead retraining from scratch |

### 5.2 Accuracy Validation Protocol

For any optimization that affects numerical output:

```bash
# Step 1: Run baseline on reference frames
python jetson_bcs_demo.py --video reference_100frames.mp4 \
    --out baseline.mp4 --head production_head_vits.pt \
    --config production_config.json --max-frames 100 \
    --csv-labels baseline_scores.csv

# Step 2: Run optimized version
python bcs_optimized.py --video reference_100frames.mp4 \
    --out optimized.mp4 --head production_head_vits.pt \
    --config production_config.json --max-frames 100 \
    --csv-labels optimized_scores.csv

# Step 3: Compare per-cow scores
python compare_scores.py --baseline baseline_scores.csv --optimized optimized_scores.csv
```

**Acceptance criteria:**
- **Critical deployment:** QWK (quadratic weighted kappa) drop < 0.02 vs baseline
- **Standard deployment:** QWK drop < 0.05 vs baseline
- **Development/testing:** QWK drop < 0.10 vs baseline

---

## 6. Implementation Priority Matrix

### 6.1 Quadrant Chart

```
                      HIGH IMPACT
                          │
                          │
       PLAN STRATEGICALLY │  DO FIRST
       ┌──────────────────┼──────────────────┐
       │                  │                  │
       │  QNN CDSP        │  Frame skip      │
       │  INT8 quant      │  ↓ resolution    │
   H   │  Model subst.    │  DINOv2 168×168  │
   I   │  Distillation    │  YOLO conf tune  │
   G   │  Tensor decom.   │  Remove masking  │
   H   │                  │  Batch DINOv2    │
       │                  │                  │
   E   ├──────────────────┼──────────────────┤
   F   │                  │                  │
   F   │  SKIP            │  DO IF EASY      │
   O   │                  │                  │
   R   │  C++ rewrite     │  BcsHead→NumPy   │
   T   │  Custom trainer  │  FP16 ONNX       │
       │  Full retrain    │  Perf counters   │
       │                  │                  │
       └──────────────────┴──────────────────┘
                      LOW EFFORT
```

### 6.2 Recommended Execution Order

```
Priority 1 (DO FIRST) — Complete by end of Day 1:
  1. Frame skipping
  2. Reduce input resolution to 1280×720
  3. Reduce DINOv2 input to 168×168
  4. YOLO confidence threshold to 0.5
  5. Remove background masking

Priority 2 (DO IF EASY) — Complete by Day 2:
  6. BcsHead → NumPy conversion
  7. Convert models to ONNX
  8. ONNX Runtime with threaded execution

Priority 3 (PLAN STRATEGICALLY) — Complete by Week 2:
  9. QNN SDK installation and CDSP deployment
  10. INT8 quantization via QNN
  11. Multi-threaded pipeline architecture
  12. GStreamer hardware decode

Priority 4 (EVALUATE / SKIP) — Month 1+:
  13. Model substitution (TinyViT, YOLOv10)
  14. Tensor decomposition / pruning
  15. Custom lightweight feature extractor
  16. C++ rewrite with Hexagon SDK
```

---

## 7. Risk Assessment

### 7.1 Risk Matrix

| Risk | Probability | Severity | Impact | Mitigation |
|---|---|---|---|---|
| **QNN SDK installation failure** | High (40%) | Critical | Cannot use CDSP/GPU acceleration | Have CPU ONNX Runtime fallback ready. Test SDK on Ubuntu 22.04 first. |
| **INT8 accuracy collapse for DINOv2** | Medium (30%) | High | Feature quality degrades → wrong BCS scores | Always validate against FP16 baseline. Use per-tensor quantization. |
| **CDSP thermal throttling** | Medium (25%) | Medium | FPS drops after 10–15 min sustained load | Monitor temperature. Implement duty-cycling or frame skipping during high load. |
| **Memory pressure (88 MB DINOv2 + buffers)** | Low (15%) | High | OOM crashes with multiple video buffers | Use frame queue depth limit (maxsize=2). Monitor RSS via `/proc/self/status`. |
| **ONNX Runtime thread contention** | Medium (20%) | Medium | Sub-linear scaling with thread count | Pin threads to cores. Use `taskset` to isolate A78 cores for inference. |
| **Adreno GPU compute driver missing** | High (50%) | Medium | Cannot use GPU backend | CDSP is primary target; GPU is "nice to have." |
| **DINOv2 168×168 accuracy drop** | Low (15%) | Medium | Reduced BCS scoring accuracy | Validate with labeled test set. Fine-tune BcsHead for 168×168 if needed. |
| **Pipeline thread safety bugs** | Medium (20%) | Medium | Race conditions, crashes | Use `queue.Queue` (thread-safe). Add timeout to `get()` to avoid deadlocks. |
| **Qualcomm SDK license restrictions** | Low (10%) | Low | Cannot distribute QNN binaries | CPU-only fallback is always available. Check SDK EULA for redistribution terms. |

### 7.2 Platform-Specific Risks for RB3gen2

| Risk | Detail |
|---|---|
| **fastrpc permissions** | CDSP access via fastrpc may require `root` or `qualcomm` group membership. Test: `ls -l /dev/adsprpc-smd /dev/adsprpc-fastrpc` |
| **Adreno 618 OpenCL** | RB3gen2 GPU driver may not expose OpenCL compute — only OpenGL ES 3.2. Check: `clinfo` (if installed) or `ls /dev/dri/` |
| **Kernel version** | QNN SDK requires kernel 5.10+ with specific DMA-BUF and ION support. Check: `uname -r` |
| **glibc compatibility** | QNN SDK ships with prebuilt binaries linked against specific glibc. `ldd qnn-net-run` to verify linkage |
| **4×A78 vs 4×A55** | Inference must be pinned to A78 cores. Default scheduling may place tasks on A55. Use `sched_setaffinity` or `taskset`. |

### 7.3 Fallback Strategy

```
QNN SDK not available / fails to install?
  └── ONNX Runtime with 6 CPU threads (target: 15–25 FPS)

ONNX Runtime insufficient?
  └── Model size reduction + frame skipping more aggressively

Still not meeting 25 FPS?
  └── Accept 15–20 FPS as "good enough" — cows don't move fast
  └── Or upgrade to RB5/RB6 with dedicated NPU
```

---

## 8. Model Conversion & Deployment Workflow

### 8.1 ONNX Conversion Pipeline

```
PyTorch (.pt/.pth)
    │
    ├── torch.onnx.export(...)
    │       │
    │       ▼
    │   FP32 ONNX  ──── ONNX Runtime (CPU)
    │       │
    │       ├── onnxruntime.quantization.quantize_dynamic(...)
    │       │       │
    │       │       ▼
    │       │   INT8 ONNX  ──── ONNX Runtime (CPU, reduced precision)
    │       │
    │       └── onnxruntime.quantization.quant_qat(...)
    │               │
    │               ▼
    │           QAT ONNX  ──── ONNX Runtime (higher accuracy INT8)
    │
    ├── qnn-converter ...
    │       │
    │       ▼
    │   QNN Cached Binary (.bin)
    │       │
    │       ├── QNN HTP (Hexagon CDSP)  ◄── PRIMARY TARGET
    │       └── QNN GPU (Adreno/OpenCL)  ◄── IF AVAILABLE
    │
    └── TensorRT (for reference — not available on RB3gen2)
```

### 8.2 YOLOv8n-seg Conversion Steps

```bash
# Step 1: Export to ONNX (using ultralytics native exporter)
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt')
model.export(format='onnx', imgsz=640, opset=17)
# Creates: yolov8n-seg.onnx (14 MB FP32)
"

# Step 2: (Optional) FP16 via onnxconverter-common
python -m onnxconverter_common.float16_conversion \
    --input yolov8n-seg.onnx \
    --output yolov8n-seg.fp16.onnx

# Step 3: (Optional) INT8 via ONNX Runtime quantization
python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('yolov8n-seg.onnx', 'yolov8n-seg.int8.onnx',
                 weight_type=QuantType.QInt8)
"

# Step 4: Verify with ONNX Runtime
python -c "
import onnxruntime as ort
sess = ort.InferenceSession('yolov8n-seg.onnx')
print(f'Inputs: {[i.name for i in sess.get_inputs()]}')
print(f'Outputs: {[o.name for o in sess.get_outputs()]}')
"
```

### 8.3 DINOv2 ViT-S/14 Conversion Steps

```bash
# Step 1: Export to ONNX with dynamic batch
python -c "
import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'dinov2_vits14.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=17)
# Creates: dinov2_vits14.onnx (88 MB FP32)
"

# Step 2: (Strongly recommended) Reduce input size
# Modify the export script to accept a 168×168 input:
dummy = torch.randn(1, 3, 168, 168)
torch.onnx.export(model, dummy, 'dinov2_vits14_168.onnx', ...)
# Creates: dinov2_vits14_168.onnx (88 MB, but runs ~1.7× faster)
```

### 8.4 QNN SDK Deployment

```bash
# Prerequisites: Qualcomm QNN SDK 2.18+ installed in $QNN_SDK_ROOT

# Step 1: Convert DINOv2 to QNN
${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-converter \
    --input_network dinov2_vits14.onnx \
    --input_dtype float32 \
    --output_path dinov2_vits14.qnn \
    --input_tensor input,1,3,224,224 \
    --output_tensor output \
    --precision fp16

# Step 2: Quantize to INT8 (requires calibration data)
${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-quantize \
    --model dinov2_vits14.qnn \
    --calibration_dataset ./cow_calib_data/ \
    --activation_bitwidth 8 \
    --weight_bitwidth 8 \
    --output_path dinov2_vits14.int8.qnn

# Step 3: Deploy on target (RB3gen2)
# Transfer .qnn or .bin to target, then:
qnn-net-run --model dinov2_vits14.qnn --backend libQnnHtp.so \
    --input_list input_list.txt --output_dir ./outputs/
```

---

## 9. Benchmarking Methodology

### 9.1 Performance Metrics

| Metric | Definition | Target |
|---|---|---|
| **End-to-end FPS** | `total_frames / wall_clock_time` | ≥25 FPS |
| **YOLO latency** | Per-frame YOLO inference time (ms) | <50 ms |
| **DINOv2 latency** | Per-frame DINOv2 time for all cows (ms) | <30 ms |
| **Pipeline latency** | Time from frame capture to annotated output (ms) | <100 ms |
| **CPU utilization** | % across all cores | <80% steady-state |
| **Memory RSS** | Resident set size | <4 GB |

### 9.2 Benchmarking Script

```python
"""benchmark_optimization.py — Run and log optimization benchmarks."""
import time, json, csv, argparse, subprocess, os

BENCHMARK_VIDEO = "sample_500frames.mp4"
RESULTS_FILE = "benchmark_results.json"

CONFIGS = [
    {"name": "baseline", "args": "--conf 0.25 --resize 2560x1440 --dino-size 224 --skip 1 --no-mask"},
    {"name": "tier1_quick", "args": "--conf 0.5 --resize 1280x720 --dino-size 168 --skip 2 --no-mask"},
    {"name": "tier2_onnx", "args": "--conf 0.5 --resize 1280x720 --dino-size 168 --skip 2 --no-mask --backend onnx"},
    {"name": "tier3_qnn", "args": "--conf 0.5 --resize 1280x720 --dino-size 168 --skip 2 --no-mask --backend qnn"},
]

def run_benchmark(config):
    cmd = f"python bcs_benchmark.py --video {BENCHMARK_VIDEO} {config['args']} --max-frames 500"
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start
    # Parse FPS from stdout
    for line in result.stdout.split('\n'):
        if 'FPS' in line:
            fps = float(line.split()[0])
    return {"fps": fps, "wall_time": elapsed}

if __name__ == "__main__":
    results = {}
    for cfg in CONFIGS:
        print(f"Benchmarking: {cfg['name']}...")
        results[cfg['name']] = run_benchmark(cfg)
    json.dump(results, open(RESULTS_FILE, 'w'), indent=2)
    print(f"Results saved to {RESULTS_FILE}")
```

### 9.3 Profiling with Perfetto

For detailed tracing on RB3gen2:

```bash
# Record 10 seconds of pipeline execution with perfetto
adb shell perfetto -o /data/misc/perfetto-traces/bcs_trace.perfetto-trace \
    -t 10s sched freq idle gpu_mem

# Pull and open in ui.perfetto.dev
adb pull /data/misc/perfetto-traces/bcs_trace.perfetto-trace .
```

---

## 10. Monitoring & Maintenance

### 10.1 Runtime Monitoring Dashboard

Add to the pipeline for continuous monitoring:

```python
import psutil, time, json

class PerformanceMonitor:
    def __init__(self, log_interval=60):
        self.log_interval = log_interval
        self.last_log = time.time()
        self.metrics = []

    def log(self, fps, yolo_ms, dino_ms, cows_detected):
        now = time.time()
        cpu = psutil.cpu_percent(interval=0.1, percpu=True)
        mem = psutil.virtual_memory()
        entry = {
            "timestamp": now,
            "fps": fps,
            "yolo_ms": yolo_ms,
            "dino_ms": dino_ms,
            "cows": cows_detected,
            "cpu_per_core": cpu,
            "mem_used_gb": mem.used / 1e9,
            "mem_percent": mem.percent,
        }
        self.metrics.append(entry)
        if now - self.last_log >= self.log_interval:
            with open("perf_log.json", "a") as f:
                json.dump(entry, f)
                f.write("\n")
            self.last_log = now
```

### 10.2 Thermal Throttling Detection

```python
def check_thermal_throttle():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = int(f.read().strip()) / 1000  # millidegrees → degrees
        if temp > 85:
            print(f"[WARN] Thermal throttle: {temp}°C")
            return True
        return False
    except FileNotFoundError:
        return False
```

### 10.3 Regression Test Suite

```bash
# Create weekly regression test
cat > regression_test.sh << 'EOF'
#!/bin/bash
set -e
REFERENCE_DIR="test_reference_20260720"

echo "=== BCS Optimization Regression Test ==="
echo "Running 5 configurations against reference..."

for cfg in baseline tier1_quick tier2_onnx; do
    echo -n "$cfg: "
    python compare_scores.py \
        --baseline "${REFERENCE_DIR}/${cfg}_scores.csv" \
        --optimized "current_${cfg}_scores.csv" \
        --output "diff_${cfg}.json"
    
    qwk_diff=$(python -c "import json; d=json.load(open('diff_${cfg}.json')); print(d.get('qwk_diff', 999))")
    if (( $(echo "$qwk_diff < 0.02" | bc -l) )); then
        echo "PASS (QWK diff: $qwk_diff)"
    else
        echo "FAIL (QWK diff: $qwk_diff > 0.02)"
        exit 1
    fi
done

echo "=== All tests PASSED ==="
EOF
chmod +x regression_test.sh
```

---

## Appendix

### A.1 Hardware Comparison

| Feature | Jetson Orin NX (original) | Qualcomm RB3gen2 (target) | Delta |
|---|---|---|---|
| GPU | 1024-core Ampere @ 1.1 GHz | Adreno 618 @ 800 MHz | Jetson has → dedicated compute |
| Tensor Core | Yes (32) | No | Major delta |
| NPU | No | CDSP (limited) | Different paradigms |
| CUDA/TensorRT | Yes (mature) | No | Jetson advantage |
| QNN SDK | N/A | Yes (but complex) | RB3gen2 advantage |
| RAM | 8 GB LPDDR5 | 7.1 GB LPDDR4X | Comparable |
| CPU | 12-core (4×A78@2.0+8×A78@1.3) | 8-core (4×A78@2.4+4×A55@1.8) | RB3 has faster A78 |
| Power | 15–40W | ~15W | RB3 is more power-efficient |

### A.2 Glossary

| Term | Definition |
|---|---|
| BCS | Body Condition Scoring — assessing cow fatness on a 3-point scale (thin/ideal/fat) |
| CDSP | Compute Digital Signal Processor — Qualcomm's general-purpose DSP for ML inference |
| QNN | Qualcomm Neural Networks SDK — toolchain for deploying models on Qualcomm hardware |
| HTP | Hexagon Tensor Processor — the compute array inside Hexagon DSP |
| QWK | Quadratic Weighted Kappa — primary accuracy metric for BCS (ordinal classification) |
| ONNX | Open Neural Network Exchange — cross-platform model format |
| ViT-S/14 | Vision Transformer Small with 14×14 patch size |
| fastrpc | Qualcomm's inter-processor communication mechanism for CDSP access |

### A.3 References

- Qualcomm QNN SDK Documentation: `docs.qualcomm.com/browse/qnn-sdk`
- ONNX Runtime Performance Tuning: `onnxruntime.ai/docs/performance`
- DINOv2 GitHub: `github.com/facebookresearch/dinov2`
- Ultralytics YOLOv8 Export: `docs.ultralytics.com/modes/export`
- RB3gen2 Technical Reference Manual (Qualcomm Doc. 80-PH297-1)
- BCS Model Training Guide: `docs/03_bcs_training_guide.md` (project docs)
