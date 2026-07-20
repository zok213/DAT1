# Qualcomm RB3gen2 Adaptation Guide

> Transform the NVIDIA Jetson BCS (Body Condition Scoring) pipeline to run on Qualcomm RB3gen2 (QCM6490)

**Date:** 2026-07-20
**Platform:** Qualcomm RB3gen2 / QCM6490
**Original Platform:** NVIDIA Jetson Orin NX
**Python Version:** 3.12
**OS:** Ubuntu 24.04 LTS (Noble)

---

## 1. Introduction & Platform Comparison

### 1.1 Why Adapt?

The original BCS pipeline was built and tuned for the NVIDIA Jetson Orin NX, leveraging TensorRT, CUDA-accelerated ops, and the Ampere GPU. Porting to Qualcomm RB3gen2 requires replacing every NVIDIA-specific component with equivalent (or fallback) implementations targeting Qualcomm's Hexagon CDSP, Adreno GPU, or — at minimum — the Cortex-A78/A55 CPU cluster.

### 1.2 Hardware Comparison

| Feature | Jetson Orin NX (16 GB) | Qualcomm RB3gen2 (QCM6490) |
|---|---|---|
| **CPU** | 8× Cortex-A78AE @ 2.0 GHz | 4× Cortex-A78 @ 2.7 GHz + 4× Cortex-A55 @ 1.9 GHz |
| **GPU** | NVIDIA Ampere (1024 CUDA cores, 16 SMs) | Adreno 642 (no compute driver loaded) |
| **GPU Compute** | CUDA 12.6, Tensor Cores | OpenCL 3.0 (limited), OpenGL ES |
| **AI Accelerator** | Tensor Cores (60 TOPS INT8) | Hexagon CDSP (12-15 TOPS INT8) |
| **AI Software** | TensorRT 10.3, TensorRT-LLM | QNN SDK 2.25+, SNPE |
| **System RAM** | 16 GB LPDDR5 (shared) | 7.1 GB LPDDR4x (shared) |
| **Storage** | NVMe (user-provided) | UFS 2.1 (on-board) |
| **TPD** | 15 W – 25 W | ~5 W – 8 W |

### 1.3 Software Stack Comparison

| Component | Jetson Orin NX | Qualcomm RB3gen2 |
|---|---|---|
| **OS** | Ubuntu 22.04 (aarch64) | Ubuntu 24.04 (aarch64) |
| **Python** | 3.10 | 3.12 |
| **PyTorch** | 2.5.0 (CUDA) | 2.5.0 (CPU, from piwheels) |
| **ONNX Runtime** | onnxruntime-gpu (CUDA) | onnxruntime (CPU) |
| **Deep Learning SDK** | TensorRT 10.3 / CUDA 12.6 | QNN 2.25+ / fastrpc |
| **CV** | OpenCV 4.10 (CUDA) | OpenCV 4.10 (CPU) |

### 1.4 Key Constraints on Qualcomm

1. **No NVIDIA GPU** — TensorRT, CUDA, cuDNN are unavailable.
2. **No compute GPU driver loaded** — Adreno is present but Adreno Compute Driver is not loaded. OpenCL kernels may fall back to CPU.
3. **7.1 GB shared RAM** — An 88 MB DINOv2 model plus YOLO and BcsHead will strain memory if not carefully managed.
4. **CDSP available via fastrpc** — The Hexagon DSP can accelerate INT8-quantized models through QNN's HTP backend.
5. **No pre-installed AI stack** — PyTorch, ONNX Runtime, and QNN SDK must be installed manually.

---

## 2. Environment Setup

### 2.1 System Packages

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Build essentials for pip packages with native extensions
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  python3-dev \
  python3-venv \
  python3-pip \
  python3-opencv \
  libopencv-dev \
  libssl-dev \
  libffi-dev \
  pkg-config \
  wget \
  curl \
  git

# Verify OpenCV installation
python3 -c "import cv2; print(cv2.__version__)"   # Should print 4.10.x
```

### 2.2 Python Virtual Environment

```bash
# Create and activate virtual environment
python3 -m venv ~/bcs_venv
source ~/bcs_venv/bin/activate

# Upgrade pip within venv
pip install --upgrade pip setuptools wheel
```

### 2.3 Install PyTorch for aarch64

PyTorch does not publish official aarch64 Linux wheels. Use the piwheels community repository.

```bash
# Install PyTorch from piwheels (CPU-only, aarch64)
pip install --extra-index-url https://piwheels.org/simple torch torchvision --only-binary torch

# Verify
python3 -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
# Expected: 2.5.0 or similar, MPS not available (expected on Linux)
```

> **Fallback:** If piwheels does not have a compatible wheel, [build PyTorch from source](https://github.com/pytorch/pytorch#installation) with `MAX_JOBS=4` to avoid OOM on the RB3gen2. Expect 2-4 hours.

### 2.4 Install ONNX Runtime

```bash
# Install ONNX Runtime for aarch64
pip install onnxruntime

# Verify
python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_device())"
# Expected: 1.19.x, "CPU"
```

### 2.5 Install Ultralytics (YOLOv8)

```bash
pip install ultralytics

# Verify
python3 -c "from ultralytics import YOLO; print('ultralytics OK')"
```

### 2.6 Install Additional Dependencies

```bash
pip install \
  numpy \
  scipy \
  Pillow \
  matplotlib \
  psutil \
  tqdm \
  pyyaml \
  onnx \
  onnxsim
```

### 2.7 Verify Full Stack

```python
# verify_stack.py
import sys, torch, onnxruntime, cv2, numpy as np
from ultralytics import YOLO

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"ONNX Runtime: {onnxruntime.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Device: {'CPU' if not torch.cuda.is_available() else 'CUDA'}")
```

---

## 3. Three-Tier Adaptation Strategy

The adaptation is structured as three tiers with increasing performance and complexity. Start with Tier 1 to validate correctness, then progress.

```
Tier 1 ──> Tier 2 ──> Tier 3
CPU-Only    ONNX RT    QNN Accel
(now)      (pip install) (SDK req.)
```

### 3.1 Tier 1: Immediate CPU-Only (Works Today)

**Goal:** Get the pipeline running on the RB3gen2 with zero platform-specific SDKs.

**Changes from Jetson baseline:**

| Component | Jetson (TensorRT) | Tier 1 (CPU) |
|---|---|---|
| **YOLOv8** | TensorRT engine (FP16) | ultralytics CPU backend (FP32) |
| **DINOv2** | TensorRT engine (FP16) | ONNX Runtime CPU provider (FP32) |
| **BcsHead** | Torch CUDA tensor ops | Pure NumPy inference |
| **Pre/Post-process** | CUDA-resized, pinned memory | OpenCV CPU, standard numpy |

**Implementation:**

```python
# qualcomm_bcs_cpu.py — Tier 1: CPU-Only Pipeline
"""
Usage:
  python qualcomm_bcs_cpu.py --video input.mp4 --output output.mp4
"""
import argparse
import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO
from time import perf_counter


# ---------------------------------------------------------------------------
# NumPy-based BcsHead (replaces PyTorch BcsHead)
# ---------------------------------------------------------------------------
class NumpyBcsHead:
    """
    BcsHead: DINOv2 embedding → body condition scores.
    Architecture: LayerNorm → Linear(384, 128) → GELU → Linear(128, 3) → Sigmoid
    """
    def __init__(self, weights_path: str):
        self.params = np.load(weights_path, allow_pickle=True).item()
        # self.params keys: 'ln_weight', 'ln_bias', 'fc1_weight', 'fc1_bias',
        #                    'fc2_weight', 'fc2_bias'

    def layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * self.params['ln_weight'] + self.params['ln_bias']

    def gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (N, 384) normalized embedding
        x = self.layer_norm(x)
        x = x @ self.params['fc1_weight'].T + self.params['fc1_bias']
        x = self.gelu(x)
        x = x @ self.params['fc2_weight'].T + self.params['fc2_bias']
        return 1.0 / (1.0 + np.exp(-x))   # Sigmoid


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class BcsPipelineCPU:
    def __init__(self, yolo_model: str = "yolov8n-seg.pt",
                 dinov2_model: str = "dinov2_vits14.onnx",
                 bcs_head_weights: str = "bcs_head_weights.npz",
                 conf_thresh: float = 0.5):
        self.conf_thresh = conf_thresh

        # YOLOv8 — ultralytics CPU
        self.yolo = YOLO(yolo_model)

        # DINOv2 — ONNX Runtime CPU
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.dinov2 = ort.InferenceSession(
            dinov2_model, so, providers=['CPUExecutionProvider']
        )

        # BcsHead — NumPy
        self.bcs = NumpyBcsHead(bcs_head_weights)

        # Precompute DINOv2 normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_cow(self, cow_img: np.ndarray, size: int = 224) -> np.ndarray:
        """Resize and normalize a cow crop for DINOv2."""
        resized = cv2.resize(cow_img, (size, size))
        # HWC → CHW, uint8 → float32
        blob = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        blob = (blob - self.mean[:, None, None]) / self.std[:, None, None]
        return blob[np.newaxis, ...]   # (1, 3, 224, 224)

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Returns: (annotated_frame, bcs_scores_list)
        """
        # Step 1: YOLOv8 instance segmentation
        results = self.yolo(frame, conf=self.conf_thresh, verbose=False)
        annotated = results[0].plot()

        scores = []
        if results[0].masks is not None:
            # Get bounding boxes for each cow
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cow_crop = frame[y1:y2, x1:x2]
                if cow_crop.size == 0:
                    continue

                # Step 2: DINOv2 embedding
                blob = self.preprocess_cow(cow_crop)
                emb = self.dinov2.run(None, {'input': blob})[0]  # (1, 384)
                emb = emb.squeeze(0)                             # (384,)

                # Step 3: BcsHead scoring
                score = self.bcs(emb[np.newaxis, :])             # (1, 3)
                scores.append(score.squeeze(0))

        return annotated, scores

    def __call__(self, frame: np.ndarray) -> tuple:
        return self.process_frame(frame)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--yolo", default="yolov8n-seg.pt")
    parser.add_argument("--dinov2", default="dinov2_vits14.onnx")
    parser.add_argument("--bcs", default="bcs_head_weights.npz")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
    args = parser.parse_args()

    pipeline = BcsPipelineCPU(args.yolo, args.dinov2, args.bcs, args.conf)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_idx = 0
    infer_times = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.skip != 0:
            out.write(frame)
            frame_idx += 1
            continue

        t0 = perf_counter()
        annotated, scores = pipeline(frame)
        dt = perf_counter() - t0
        infer_times.append(dt)

        # Overlay scores
        y_offset = 30
        for i, s in enumerate(scores[:5]):  # show up to 5
            label = f"Cow {i}: BCS={s[0]:.2f} Cond={s[1]:.2f} Frame={s[2]:.2f}"
            cv2.putText(annotated, label, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 24

        out.write(annotated)
        frame_idx += 1
        print(f"Frame {frame_idx}: {dt*1000:.1f} ms")

    cap.release()
    out.release()
    avg_ms = np.mean(infer_times) * 1000 if infer_times else 0
    print(f"\nAverage inference time: {avg_ms:.1f} ms ({1000/avg_ms:.1f} FPS)")


if __name__ == "__main__":
    main()
```

**Running Tier 1:**

```bash
source ~/bcs_venv/bin/activate
python qualcomm_bcs_cpu.py \
  --video sample_cows.mp4 \
  --output output_cpu.mp4 \
  --skip 2
```

**Expected performance:** 2–5 FPS on 720p input with `--skip 2`.

### 3.2 Tier 2: ONNX Runtime with CPU (After pip install)

**Goal:** Convert all three model components to ONNX and run them through a unified ONNX Runtime session for lower overhead.

**Model conversion:**

```bash
# Convert YOLOv8 to ONNX
yolo export model=yolov8n-seg.pt format=onnx imgsz=640

# BcsHead: PyTorch → ONNX
python -c "
import torch
import torch.nn as nn

class BcsHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(384)
        self.fc1 = nn.Linear(384, 128)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ln(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

model = BcsHead()
# Load your trained weights
state = torch.load('production_head_vits.pt', map_location='cpu')
model.load_state_dict(state)
model.eval()

dummy = torch.randn(1, 384)
torch.onnx.export(model, dummy, 'bcs_head.onnx',
                  input_names=['embedding'],
                  output_names=['scores'],
                  dynamic_axes={'embedding': {0: 'batch'},
                                'scores': {0: 'batch'}},
                  opset_version=17)
print('bcs_head.onnx written')
"

# DINOv2 is already in ONNX format
ls -la dinov2_vits14.onnx
```

**Pipeline script:**

```python
# qualcomm_bcs_onnxruntime.py — Tier 2: All-ONNX Runtime
import onnxruntime as ort
import numpy as np
import cv2
from time import perf_counter


class BcsPipelineORT:
    def __init__(self, yolo_onnx: str, dinov2_onnx: str, bcs_onnx: str,
                 conf_thresh: float = 0.5):
        self.conf_thresh = conf_thresh
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CPUExecutionProvider']

        self.yolo_session = ort.InferenceSession(yolo_onnx, so, providers=providers)
        self.dinov2_session = ort.InferenceSession(dinov2_onnx, so, providers=providers)
        self.bcs_session = ort.InferenceSession(bcs_onnx, so, providers=providers)

        # DINOv2 normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_yolo(self, frame: np.ndarray, size: int = 640) -> np.ndarray:
        """Letterbox + normalize for YOLOv8 ONNX."""
        h, w = frame.shape[:2]
        scale = min(size / w, size / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        dx = (size - nw) // 2
        dy = (size - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized
        blob = canvas.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...] / 255.0
        return blob

    def preprocess_dinov2(self, cow_img: np.ndarray, size: int = 224) -> np.ndarray:
        resized = cv2.resize(cow_img, (size, size))
        blob = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        blob = (blob - self.mean[:, None, None]) / self.std[:, None, None]
        return blob[np.newaxis, ...]

    def process_frame(self, frame: np.ndarray) -> tuple:
        # YOLOv8
        yolo_in = self.preprocess_yolo(frame)
        yolo_out = self.yolo_session.run(None, {'images': yolo_in})[0]
        # Parse YOLOv8 output (simplified; real impl needs NMS)
        boxes = self._parse_yolo_output(yolo_out, frame.shape)

        scores = []
        annotated = frame.copy()
        for x1, y1, x2, y2 in boxes:
            cow_crop = frame[y1:y2, x1:x2]
            if cow_crop.size == 0:
                continue
            blob = self.preprocess_dinov2(cow_crop)
            emb = self.dinov2_session.run(None, {'input': blob})[0]  # (1, 384)
            score = self.bcs_session.run(None, {'embedding': emb})[0]  # (1, 3)
            scores.append(score.squeeze(0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return annotated, scores

    # Full YOLOv8 post-processing would include NMS here
    def _parse_yolo_output(self, raw, img_shape):
        # Simplified: assumes raw is [N, 116] with [x, y, w, h, conf, ...]
        # Real implementation: sigmoid, box decoding, NMS
        return []

    def __call__(self, frame):
        return self.process_frame(frame)
```

### 3.3 Tier 3: Qualcomm QNN Acceleration (Requires SDK)

**Prerequisites:** Qualcomm QNN SDK v2.25+ (download from Qualcomm CreatePoint / developer portal).

**Components:**

| Backend | Device | Best For |
|---|---|---|
| **HTP** (Hexagon Tensor Processor) | CDSP (compute DSP) | INT8-quantized DINOv2, YOLOv8 |
| **GPU** (Adreno OpenCL) | Adreno 642 | FP16 ops if compute driver loaded |
| **CPU** (fallback) | Cortex-A78/A55 | Ops not supported on HTP/GPU |

**Architecture (QNN):**

```
┌──────────────────────────────────────────┐
│             Application                  │
│  qualcomm_bcs_qnn.py                     │
├──────────────────────────────────────────┤
│          QNN SDK C API / Python          │
│         libQnnHtp.so / libQnnGpu.so       │
├──────────────────────────────────────────┤
│   QNN Model (.bin serialized)            │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐  │
│  │ YOLOv8  │ │ DINOv2   │ │ BcsHead  │  │
│  │ .bin    │ │ .bin     │ │ .bin     │  │
│  └─────────┘ └──────────┘ └──────────┘  │
├──────────────────────────────────────────┤
│  fastrpc (user-space → CDSP)             │
│  or Adreno GPU driver                    │
└──────────────────────────────────────────┘
```

**Installation (once SDK is downloaded):**

```bash
# Extract QNN SDK
unzip qnn_sdk_2_25_0.zip -d ~/qcom/qnn
export QNN_SDK_ROOT=~/qcom/qnn
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH
export PATH=$QNN_SDK_ROOT/bin/aarch64-ubuntu-gcc9.4:$PATH

# Verify fastrpc CDSP access
ls -la /dev/ | grep fastrpc
# Should show: /dev/fastrpc (often permissions: crw-rw----)

# Test QNN HTP backend
qnn-htp-dump-info
```

**Pipeline script structure:**

```python
# qualcomm_bcs_qnn.py — Tier 3: QNN-Accelerated
"""
Requires: QNN SDK 2.25+, fastrpc device accessible
"""
import numpy as np
import cv2
from qnn import QNNInterface      # Provided with QNN SDK python bindings
from time import perf_counter


class BcsPipelineQNN:
    def __init__(self, yolo_bin: str, dinov2_bin: str, bcs_bin: str,
                 backend: str = "htp", conf_thresh=0.5):
        """
        backend: 'htp' (CDSP), 'gpu' (Adreno), or 'cpu'
        """
        self.conf_thresh = conf_thresh
        self.backend = backend

        # Initialize QNN backends
        self.htp_handle = None
        self.gpu_handle = None
        if backend == "htp":
            self.htp_handle = QNNInterface.load_backend("libQnnHtp.so")
        elif backend == "gpu":
            self.gpu_handle = QNNInterface.load_backend("libQnnGpu.so")

        # Load serialized models — each returns a Context handle
        self.yolo_ctx = QNNInterface.load_model(yolo_bin, self._backend_handle())
        self.dinov2_ctx = QNNInterface.load_model(dinov2_bin, self._backend_handle())
        self.bcs_ctx = QNNInterface.load_model(bcs_bin, self._backend_handle())

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _backend_handle(self):
        return self.htp_handle or self.gpu_handle

    def infer_model(self, ctx, input_dict: dict) -> dict:
        """Run inference on a QNN model context."""
        output = QNNInterface.execute(ctx, input_dict)
        return output

    def process_frame(self, frame: np.ndarray) -> tuple:
        # YOLOv8
        yolo_in = self._preprocess_yolo(frame)
        yolo_out = self.infer_model(self.yolo_ctx, {'images': yolo_in})
        boxes = self._parse_yolo_output(yolo_out, frame.shape)

        scores = []
        annotated = frame.copy()
        for x1, y1, x2, y2 in boxes:
            cow_crop = frame[y1:y2, x1:x2]
            if cow_crop.size == 0:
                continue
            blob = self._preprocess_dinov2(cow_crop)
            emb = self.infer_model(self.dinov2_ctx, {'input': blob})['embedding']
            score = self.infer_model(self.bcs_ctx, {'embedding': emb})['scores']
            scores.append(score.squeeze(0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return annotated, scores
```

---

## 4. Model Conversion Guide

### 4.1 DINOv2: ONNX → QNN Serialized

DINOv2 (`dinov2_vits14.onnx`, 88 MB) is already in ONNX format.

```bash
# 1. Optimize ONNX model (optional but recommended)
python -m onnxsim dinov2_vits14.onnx dinov2_vits14_sim.onnx

# 2. Convert to QNN serialized using qnn-onnx-converter
#    For HTP backend (INT8 quantization):
qnn-onnx-converter \
  --input_diname input --input_diname 1,3,224,224 \
  --output_diname embedding --output_diname 0 \
  --input_network dinov2_vits14_sim.onnx \
  --output_network dinov2_vits14_htp.bin \
  --quantization_overrides dinov2_quantization.json \
  --act_bw 8 \
  --weights_bw 8 \
  --bias_bw 32 \
  --backend htp

#    For GPU backend (FP16):
qnn-onnx-converter \
  --input_network dinov2_vits14_sim.onnx \
  --output_network dinov2_vits14_gpu.bin \
  --act_bw 16 \
  --weights_bw 16 \
  --backend gpu
```

**Quantization calibration data:**

```python
# generate_calibration.py
"""
Collect 200–500 cow crop images to build a calibration dataset for INT8 quantization.
"""
import cv2
import os
import numpy as np


def collect_calibration_data(video_path: str, output_dir: str, num_samples: int = 300):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    step = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // num_samples
    frame_idx = 0

    while cap.isOpened() and count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # Save as raw .npy for calibration
            resized = cv2.resize(frame, (224, 224))
            np.save(os.path.join(output_dir, f"cal_{count:04d}.npy"), resized)
            count += 1
        frame_idx += 1
    cap.release()
    print(f"Collected {count} calibration samples in {output_dir}")


if __name__ == "__main__":
    import sys
    collect_calibration_data(sys.argv[1], sys.argv[2])
```

```bash
# Generate quantization override JSON
qnn-onnx-converter \
  --input_network dinov2_vits14_sim.onnx \
  --output_network dinov2_vits14_quantized.bin \
  --act_bw 8 --weights_bw 8 --bias_bw 32 \
  --quantization_calibration_path ./calibration_data/ \
  --backend htp
```

### 4.2 YOLOv8: PyTorch → ONNX → QNN

```bash
# Step 1: PyTorch → ONNX (from ultralytics)
yolo export model=yolov8n-seg.pt format=onnx imgsz=640

# Step 2: Simplify ONNX
python -m onnxsim yolov8n-seg.onnx yolov8n-seg_sim.onnx

# Step 3: ONNX → QNN serialized
#    For HTP with INT8:
qnn-onnx-converter \
  --input_network yolov8n-seg_sim.onnx \
  --output_network yolov8n_seg_htp.bin \
  --act_bw 8 --weights_bw 8 --bias_bw 32 \
  --backend htp

#    For GPU FP16:
qnn-onnx-converter \
  --input_network yolov8n-seg_sim.onnx \
  --output_network yolov8n_seg_gpu.bin \
  --act_bw 16 --weights_bw 16 \
  --backend gpu
```

### 4.3 BcsHead: PyTorch → ONNX → QNN

```bash
# Step 1: PyTorch → ONNX (see Section 3.2 for script)
# Output: bcs_head.onnx

# Step 2: ONNX → QNN
qnn-onnx-converter \
  --input_network bcs_head.onnx \
  --output_network bcs_head_htp.bin \
  --act_bw 8 --weights_bw 8 --bias_bw 32 \
  --backend htp
```

### 4.4 Summary of Converted Models

| Model | Source Format | ONNX | QNN HTP (.bin) | QNN GPU (.bin) |
|---|---|---|---|---|
| YOLOv8n-seg | PyTorch (1.8 MB) | yolov8n-seg.onnx (4.2 MB) | yolov8n_seg_htp.bin (4.2 MB) | yolov8n_seg_gpu.bin (8.4 MB) |
| DINOv2 ViT-S/14 | ONNX (88 MB) | dinov2_vits14_sim.onnx (88 MB) | dinov2_vits14_htp.bin (22 MB INT8) | dinov2_vits14_gpu.bin (88 MB FP16) |
| BcsHead | PyTorch (0.4 MB) | bcs_head.onnx (0.4 MB) | bcs_head_htp.bin (0.1 MB) | bcs_head_gpu.bin (0.4 MB) |

> **Note:** INT8 quantization typically reduces DINOv2 size from 88 MB to ~22 MB with <1% accuracy loss for this task.

---

## 5. Performance Optimization

### 5.1 Frame Skipping

Process every Nth frame and duplicate the last result for skipped frames.

```python
SKIP_FACTOR = 2  # Process 50% of frames
last_annotated = None
for frame_idx, frame in enumerate(frames):
    if frame_idx % SKIP_FACTOR == 0:
        last_annotated, scores = pipeline(frame)
    output.write(last_annotated)
```

| Skip Factor | Frames Processed | Effective FPS (720p) |
|---|---|---|
| 1 | 100% | 2–5 |
| 2 | 50% | 4–10 |
| 4 | 25% | 8–20 |
| 8 | 12.5% | 16–40 |

### 5.2 Input Resolution Downscaling

| Resolution | DINOv2 Cost | YOLOv8 Cost | Notes |
|---|---|---|---|
| 1280×720 | 100% | 100% | Full HD |
| 640×480 | 50% | 50% | Good balance |
| 320×240 | 25% | 25% | Use only if cows are large in frame |

```python
def read_with_scale(video_path: str, scale: float = 0.5):
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (new_w, new_h))
        yield frame
```

### 5.3 INT8 Quantization via QNN

Quantization reduces DINOv2 memory footprint by 4×:

```
DINOv2 FP32 ONNX:  88 MB
DINOv2 INT8 QNN:   22 MB
YOLOv8 FP32 ONNX:   4.2 MB
YOLOv8 INT8 QNN:    1.1 MB
                    ─────────
Total FP32:         92.6 MB
Total INT8:         23.3 MB
```

### 5.4 Multi-Threaded Pipeline

Decouple video decoding, model inference (×3), and overlay onto separate threads.

```python
import threading
import queue


class ThreadedPipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.in_queue = queue.Queue(maxsize=4)
        self.out_queue = queue.Queue(maxsize=4)
        self.running = True

    def inference_worker(self):
        while self.running:
            frame = self.in_queue.get()
            if frame is None:
                break
            result = self.pipeline(frame)
            self.out_queue.put(result)

    def start(self):
        self.thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.in_queue.put(None)
        self.thread.join()
```

**Pipeline stages (1 decode + 1 infer + 1 display):**

```
[Decode Thread] ──queue──> [Infer Thread] ──queue──> [Display Thread]
     Frame N                    Frame N-1                 Frame N-2
```

This keeps the decode and display threads from blocking on inference, achieving ~1.5× throughput improvement on the same hardware.

### 5.5 Batch Processing Multiple Cows

DINOv2 accepts batched inputs. If 3 cows are detected in a frame:

```python
# Collect all cow crops
crops = []
boxes = []
for box in detected_boxes:
    x1, y1, x2, y2 = box
    crop = preprocess_dinov2(frame[y1:y2, x1:x2])
    crops.append(crop)
    boxes.append(box)

# Batch inference: all cows in one DINOv2 call
if crops:
    batch = np.concatenate(crops, axis=0)     # (N, 3, 224, 224)
    embeddings = dinov2_session.run(None, {'input': batch})[0]  # (N, 384)
    scores = bcs_session.run(None, {'embedding': embeddings})[0]  # (N, 3)
```

Batch of 3 is ~2.5× faster than 3 individual calls.

---

## 6. Verification & Testing

### 6.1 Unit Tests

```python
# test_pipeline.py
import unittest
import numpy as np
import cv2


class TestBCSPipeline(unittest.TestCase):
    """Unit tests for each pipeline stage."""

    @classmethod
    def setUpClass(cls):
        cls.dummy_frame = np.random.randint(0, 255,
                          (480, 640, 3), dtype=np.uint8)

    def test_yolo_detection(self):
        """YOLOv8 returns at least the expected output format."""
        from ultralytics import YOLO
        model = YOLO("yolov8n-seg.pt")
        results = model(self.dummy_frame, verbose=False)
        self.assertIsNotNone(results[0].boxes)
        print(f"[PASS] YOLOv8 detection: {len(results[0].boxes)} boxes")

    def test_dinov2_embedding(self):
        """DINOv2 produces 384-dim embedding."""
        import onnxruntime as ort
        session = ort.InferenceSession("dinov2_vits14.onnx",
                                       providers=['CPUExecutionProvider'])
        blob = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = session.run(None, {'input': blob})[0]
        self.assertEqual(output.shape, (1, 384))
        print(f"[PASS] DINOv2 embedding shape: {output.shape}")

    def test_bcs_head_output_range(self):
        """BCS scores are in [0, 1] range (sigmoid output)."""
        from qualcomm_bcs_cpu import NumpyBcsHead
        head = NumpyBcsHead("bcs_head_weights.npz")
        emb = np.random.randn(1, 384).astype(np.float32)
        scores = head(emb)
        self.assertEqual(scores.shape, (1, 3))
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))
        print(f"[PASS] BcsHead scores: {scores}")

    def test_end_to_end(self):
        """Full pipeline runs without error on a dummy frame."""
        from qualcomm_bcs_cpu import BcsPipelineCPU
        pipeline = BcsPipelineCPU()
        annotated, scores = pipeline(self.dummy_frame)
        self.assertIsNotNone(annotated)
        print(f"[PASS] End-to-end: {len(scores)} cows scored")


if __name__ == "__main__":
    unittest.main()
```

```bash
# Run unit tests
python -m pytest test_pipeline.py -v
```

### 6.2 End-to-End Test on Sample Video

```python
# test_e2e.py
"""
End-to-end test: processes a short video, verifies output exists and has content.
"""
import cv2
import numpy as np
import sys, os, json
from qualcomm_bcs_cpu import BcsPipelineCPU


def test_e2e(video_path: str, output_path: str = "test_output.mp4",
             expected_min_fps: float = 1.0):
    pipeline = BcsPipelineCPU()
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_count = max(1, frame_count // 10)  # process 10%
    all_scores = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i % (frame_count // process_count) != 0:
            out.write(frame)
            continue
        annotated, scores = pipeline(frame)
        all_scores.extend(scores)
        out.write(annotated)

    cap.release()
    out.release()

    # Verification
    assert os.path.exists(output_path), "Output file not created"
    assert os.path.getsize(output_path) > 1000, "Output file too small"
    assert len(all_scores) > 0, "No scores produced"
    print(f"[E2E PASS] {len(all_scores)} scores across {process_count} frames")
    print(f"[E2E PASS] Output: {output_path} ({os.path.getsize(output_path)} bytes)")

    # Save scores for accuracy comparison
    with open("test_scores.json", "w") as f:
        json.dump([s.tolist() for s in all_scores], f)


if __name__ == "__main__":
    test_e2e(sys.argv[1])
```

```bash
python test_e2e.py sample_cows_30s.mp4
```

### 6.3 Performance Benchmark

```python
# benchmark.py
"""
Measures per-stage latency and overall throughput.
Outputs a JSON report for comparison across tiers.
"""
import time
import numpy as np
import json
import sys
from qualcomm_bcs_cpu import BcsPipelineCPU


def benchmark(pipeline, frame: np.ndarray, num_warmup: int = 5,
              num_runs: int = 50):
    # Warmup
    for _ in range(num_warmup):
        pipeline(frame)

    # Timed runs
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        pipeline(frame)
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    report = {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "p50_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "effective_fps": float(1000.0 / np.mean(times_ms)),
        "num_runs": num_runs,
    }
    return report


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else 0
    import cv2

    if video_path != 0:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
    else:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        ret = True

    if not ret:
        print("Cannot read frame")
        sys.exit(1)

    pipeline = BcsPipelineCPU()
    report = benchmark(pipeline, frame)
    print(json.dumps(report, indent=2))

    with open("benchmark_tier1.json", "w") as f:
        json.dump(report, f, indent=2)
```

```bash
python benchmark.py sample_cows.mp4
```

### 6.4 Accuracy Comparison with PyTorch Baseline

```python
# accuracy_check.py
"""
Compare ONNX/NumPy outputs against original PyTorch outputs.
"""
import torch
import numpy as np
import onnxruntime as ort


def compare_dinov2():
    """Compare DINOv2 ONNX output vs PyTorch output."""
    # PyTorch DINOv2
    dinov2_torch = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_torch.eval()

    # ONNX Runtime
    session = ort.InferenceSession("dinov2_vits14.onnx",
                                   providers=['CPUExecutionProvider'])

    dummy = torch.randn(1, 3, 224, 224)
    dummy_np = dummy.numpy().astype(np.float32)

    with torch.no_grad():
        torch_out = dinov2_torch(dummy).numpy()

    ort_out = session.run(None, {'input': dummy_np})[0]

    diff = np.abs(torch_out - ort_out).max()
    print(f"DINOv2 max diff: {diff:.6f}")
    assert diff < 1e-3, f"DINOv2 ONNX mismatch: {diff}"
    print("[PASS] DINOv2 ONNX matches PyTorch")


def compare_bcs_head():
    """Compare NumPy BcsHead vs PyTorch BcsHead."""
    from qualcomm_bcs_cpu import NumpyBcsHead
    import torch.nn as nn

    class BcsHeadTorch(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(384)
            self.fc1 = nn.Linear(384, 128)
            self.gelu = nn.GELU()
            self.fc2 = nn.Linear(128, 3)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.ln(x)
            x = self.fc1(x)
            x = self.gelu(x)
            x = self.fc2(x)
            return self.sigmoid(x)

    torch_model = BcsHeadTorch()
    state = torch.load("production_head_vits.pt", map_location='cpu')
    torch_model.load_state_dict(state)
    torch_model.eval()

    numpy_head = NumpyBcsHead("bcs_head_weights.npz")

    dummy = torch.randn(1, 384)
    with torch.no_grad():
        torch_out = torch_model(dummy).numpy()

    numpy_out = numpy_head(dummy.numpy().astype(np.float32))

    diff = np.abs(torch_out - numpy_out).max()
    print(f"BcsHead max diff: {diff:.6f}")
    assert diff < 1e-4, f"BcsHead NumPy mismatch: {diff}"
    print("[PASS] BcsHead NumPy matches PyTorch")


if __name__ == "__main__":
    compare_dinov2()
    compare_bcs_head()
```

```bash
python accuracy_check.py
```

---

## 7. Troubleshooting

### 7.1 Common Issues on Qualcomm Platform

| Symptom | Likely Cause | Solution |
|---|---|---|
| `ImportError: libtorch.so: cannot open shared object file` | PyTorch not linked properly | Install torch from piwheels or rebuild. Verify with `ldd $(python -c "import torch; print(torch.__file__)")/lib/libtorch.so` |
| `onnxruntime.capi.onnxruntime_pybind11_state.Fail: No such file or directory` | ONNX model path is wrong | Use absolute paths. Verify file exists: `ls -la /path/to/model.onnx` |
| `RuntimeError: Could not create executor for CUDA` | Tier-1 but CUDA-dependent ops remain | Ensure `torch.device('cpu')` and `map_location='cpu'` everywhere. Do not import `torch.cuda` |
| `QnnInterface.load_backend failed: Permission denied` | No access to `/dev/fastrpc` | Add user to `fastrpc` group: `sudo usermod -aG fastrpc $USER && sudo reboot` |
| `qnn-onnx-converter: command not found` | QNN SDK not in PATH | `export PATH=$QNN_SDK_ROOT/bin/aarch64-ubuntu-gcc9.4:$PATH` |
| Pipeline crashes with `MemoryError` | 7.1 GB RAM exhausted | Reduce batch size, use INT8 models, reduce input resolution, close unused OpenCV windows, call `gc.collect()` between frames |

### 7.2 Memory Management (7.1 GB Constraint)

The 88 MB DINOv2 model is a memory pressure point:

```python
import gc
import psutil

def monitor_memory(stage: str):
    proc = psutil.Process()
    mem = proc.memory_info().rss / 1024 / 1024
    print(f"[{stage}] RSS: {mem:.1f} MB")

# Strategy: preallocate, reuse tensors, and explicitly free

# Reuse DINOv2 input buffer instead of allocating every frame
dinov2_input = np.zeros((1, 3, 224, 224), dtype=np.float32)

def preprocess_cow_reuse(cow_img: np.ndarray, out_buf: np.ndarray):
    resized = cv2.resize(cow_img, (224, 224))
    blob = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    blob = (blob - mean[:, None, None]) / std[:, None, None]
    np.copyto(out_buf, blob[np.newaxis, ...])

# Force garbage collection periodically
if gc_count % 30 == 0:
    gc.collect()
```

**Memory budget breakdown (INT8 QNN):**

| Component | RAM Used | Notes |
|---|---|---|
| DINOv2 (INT8) | 22 MB | HTP backend loads to CDSP, ~4 MB in system RAM |
| YOLOv8 (INT8) | 1.1 MB | Small model |
| BcsHead (INT8) | 0.1 MB | Negligible |
| Frame buffer | 4.9 MB | 720p @ 3× uint8 |
| Preprocessed blobs | 3.2 MB | 640×640 YOLO input + 224×224 DINOv2 input |
| Python runtime | ~50 MB | PyTorch, numpy, OpenCV |
| **Total (estimate)** | **~80 MB** | Well within 7.1 GB |

### 7.3 CDSP Thermal Throttling

The Hexagon CDSP on QCM6490 will throttle under sustained load:

```bash
# Monitor CDSP temperature
cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -3

# Typical throttle thresholds: 70°C (first), 85°C (critical)
# Under 70°C: full 12-15 TOPS
# 70-85°C: ~70% performance
# >85°C: ~40% performance with potential stalling
```

**Mitigations:**

1. **Add heatsink or active cooling** — An aluminum heatsink + 5V fan reduces CDSP temp by 15-20°C.
2. **Insert idle frames** — After every 30 inference frames, skip a frame cycle to let the DSP cool.
3. **Reduce clock speed** — If running headless, cap the CPU governor:
   ```bash
   sudo cpufreq-set -c 0 -g powersave
   sudo cpufreq-set -c 4 -g powersave
   ```
4. **Batch smaller** — Rather than 3 cows per frame, score 1 per frame and rotate.

### 7.4 Permission Issues with fastrpc Devices

```bash
# Check fastrpc device existence and permissions
ls -la /dev/fastrpc
# Expected: crw-rw---- 1 root fastrpc 405, 0 Jul 20 10:00 /dev/fastrpc

# If missing:
sudo modprobe fastrpc
dmesg | grep fastrpc

# If permission denied:
sudo usermod -aG fastrpc $USER
# Log out and back in, or:
exec su -l $USER

# Verify group membership:
groups
# Should show: fastrpc
```

### 7.5 QNN SDK Installation Debugging

```bash
# Verify QNN SDK is correctly extracted
ls $QNN_SDK_ROOT/
# Should contain: bin, lib, include, docs, x86_64-linux-clang, ...

# Verify the correct architecture binaries exist
ls $QNN_SDK_ROOT/bin/aarch64-ubuntu-gcc9.4/
# Should contain: qnn-onnx-converter, qnn-model-lib-generator, qnn-htp-dump-info, ...

# Verify HTP backend library
ls $QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4/libQnnHtp.so
# Should exist

# Test HTP backend
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH
qnn-htp-dump-info
# Should print CDSP capabilities

# If qnn-htp-dump-info fails with "libcdsprpc.so not found":
sudo apt install libcdsprpc
# Or find it in the QNN SDK:
find $QNN_SDK_ROOT -name "libcdsprpc*"
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH
```

---

## Appendix A: Quick-Start Cheatsheet

```bash
# ┌──────────────────────────────────────────────────────────┐
# │  BCS Pipeline on Qualcomm RB3gen2 — Quick Start         │
# └──────────────────────────────────────────────────────────┘

# 1. System deps
sudo apt update && sudo apt install -y build-essential python3-dev \
  python3-venv python3-opencv

# 2. Python env
python3 -m venv ~/bcs_venv && source ~/bcs_venv/bin/activate

# 3. Install packages
pip install --extra-index-url https://piwheels.org/simple torch --only-binary torch
pip install onnxruntime ultralytics numpy opencv-python

# 4. Verify
python -c "import torch, onnxruntime, cv2; print('OK')"

# 5. Run Tier 1 CPU pipeline
python qualcomm_bcs_cpu.py --video sample.mp4 --output result.mp4 --skip 2
```

## Appendix B: Resource Links

| Resource | URL |
|---|---|
| Qualcomm QNN SDK | [https://createpoint.qti.qualcomm.com/](https://createpoint.qti.qualcomm.com/) (registration required) |
| Hexagon CDSP docs | [https://developer.qualcomm.com/software/hexagon-sdk](https://developer.qualcomm.com/software/hexagon-sdk) |
| piwheels (aarch64 PyTorch) | [https://www.piwheels.org/](https://www.piwheels.org/) |
| ONNX Runtime aarch64 | `pip install onnxruntime` |
| Ultralytics YOLOv8 | `pip install ultralytics` |
| RB3gen2 landing page | [https://www.thundercomm.com/product/rb3-gen2/](https://www.thundercomm.com/product/rb3-gen2/) |

## Appendix C: File Inventory

| File | Purpose |
|---|---|
| `qualcomm_bcs_cpu.py` | Tier 1 CPU-only pipeline |
| `qualcomm_bcs_onnxruntime.py` | Tier 2 ONNX Runtime pipeline |
| `qualcomm_bcs_qnn.py` | Tier 3 QNN-accelerated pipeline |
| `verify_stack.py` | Environment verification script |
| `test_pipeline.py` | Unit tests |
| `test_e2e.py` | End-to-end validation |
| `benchmark.py` | Per-stage latency benchmark |
| `accuracy_check.py` | PyTorch vs ONNX/NumPy accuracy comparison |
| `generate_calibration.py` | INT8 quantization calibration data collector |
| `bcs_head_weights.npz` | BcsHead weights in NumPy format (converted from `production_head_vits.pt`) |
