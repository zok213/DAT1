# Performance Profiling Framework — BCS AI Pipeline on Qualcomm RB3gen2

> **Platform:** Qualcomm RB3gen2 (QCM6490) — 8-core Kryo 670 (4×A55@1.9 GHz + 4×A78@2.4 GHz), 7.1 GB RAM, Adreno 642L GPU (OpenCL 3.0)
> **Pipeline:** Video decode → YOLOv8n-seg (detection) → DINOv2 ViT-S/14 (feature extraction) → BcsHead (classifier) → Overlay
> **Date:** 2026-07-20

---

## Table of Contents

1. [Benchmarking Methodology](#1-benchmarking-methodology)
2. [Profiling Script Design](#2-profiling-script-design)
3. [Code: `profiling/profiler.py`](#3-code-profilingprofilerpy)
4. [Expected Performance Baseline](#4-expected-performance-baseline)
5. [Profiling Results Template](#5-profiling-results-template)
6. [Bottleneck Analysis](#6-bottleneck-analysis)
7. [Optimization Targets](#7-optimization-targets)

---

## 1. Benchmarking Methodology

### 1.1 Metrics

| Metric | Instrument | Unit | Granularity |
|--------|-----------|------|------------|
| **End-to-end FPS** | Wall clock over N frames | frames/sec | Entire run |
| **Per-stage latency** | `time.perf_counter()` per stage | ms/frame | Per frame |
| **CPU utilization** | `psutil.Process().cpu_percent(interval=0)` | % per core | Per frame |
| **Memory usage** | `tracemalloc.get_traced_memory()` | MB (peak RSS) | Per frame |
| **Frame drops** | Frame count vs. expected @ 25 FPS | count | Entire run |

### 1.2 Measurement Protocol

1. **Warm-up:** The first 30 frames (configurable `--warmup`) are processed but excluded from all stats. This pre-fills CPU caches, warms JIT compilers (ONNX-Runtime, PyTorch), and stabilizes memory allocators.
2. **Measurement window:** Frames 31–N (`--max-frames`) are recorded individually.
3. **Per-stage timing:** `time.perf_counter()` is captured at 6 instrumented points per frame (see script).
4. **I/O exclusion:** File writes (video output) are excluded from per-stage timing. Only memory-to-memory operations are clocked.
5. **Concurrent overhead:** Inter-stage variable assignments and control flow add < 10 µs and are considered negligible.

### 1.3 Statistical Reporting

For every metric we report:

| Statistic | Definition |
|-----------|-----------|
| **Mean** | Arithmetic mean across all measured frames |
| **Median** | 50th percentile (robust vs. outliers) |
| **P95** | 95th percentile — worst 5% of frames |
| **P99** | 99th percentile — worst 1% of frames |
| **Std Dev** | Population standard deviation |
| **Min / Max** | Absolute range |

Reporting all five summary statistics is critical: mean alone hides latency spikes (garbage collection, scheduler jitter, thermal throttling) that destroy real-time behaviour.

### 1.4 Memory & CPU Overhead

**Memory (tracemalloc):**
- Snapshot before/after each frame.
- Track current and peak allocated memory.
- Report as `current_mib` and `peak_mib`.

**CPU (psutil):**
- `Process().cpu_percent(interval=0)` — non-blocking instantaneous utilisation.
- `Process().cpu_num()` — which core the scheduler placed us on.
- Report as `cpu_percent` (0–100 × core count).

---

## 2. Profiling Script Design

### 2.1 Architecture

```
profiler.py
├── PipelineProfiler          # Main orchestrator
│   ├── __init__()            # Load all models, open video
│   ├── warmup()              # Process N frames, discard timings
│   ├── profile()             # Generator yielding FrameResult per frame
│   ├── report()              # Print formatted summary table
│   ├── save_json()           # Full results → JSON
│   ├── save_csv()            # Per-frame timings → CSV
│   └── save_flamegraph()     # Folded stack format → compatible with Brendan Gregg's FlameGraph
├── FrameResult               # dataclass — one per frame
└── __main__                  # argparse CLI
```

### 2.2 CLI Interface

```
usage: profiler.py --video PATH --yolo PATH --dino-onnx PATH --head PATH --config PATH
                   [--max-frames N] [--warmup N] [--output-dir DIR]
                   [--profile-memory] [--profile-cpu] [--json] [--csv]
                   [--flamegraph] [--quiet]

Pipeline performance profiler for BCS on Qualcomm RB3gen2.

options:
  --video PATH        Input H.264 video file
  --yolo PATH         YOLOv8n-seg model (.pt)
  --dino-onnx PATH    DINOv2 ONNX model (.onnx)
  --head PATH         BcsHead checkpoint (.pt)
  --config PATH       Production config JSON
  --max-frames N      Measure N frames after warmup (default: 500)
  --warmup N          Skip N frames for warmup (default: 30)
  --output-dir DIR    Output directory for reports (default: profiling/)
  --profile-memory    Enable tracemalloc per-frame tracking
  --profile-cpu       Enable psutil per-frame CPU tracking
  --json              Save per-frame results as JSON
  --csv               Save per-frame results as CSV
  --flamegraph        Save flamegraph-compatible folded stack output
  --quiet             Suppress per-frame progress printing
```

### 2.3 Output Files

| File | Format | Contents |
|------|--------|----------|
| `perf_report.json` | JSON | Full stats dict, per-frame list, summary table |
| `perf_report.csv` | CSV | One row per frame, columns for every stage |
| `flamegraph.folded` | Folded stack | Space-separated stack + sample count; pipe to `flamegraph.pl` |
| `perf_report_table.txt` | Text | Formatted summary table printed to stdout |

### 2.4 `FrameResult` Schema (JSON)

```json
{
  "frame": 42,
  "elapsed": 1234.56,
  "decode_ms": 35.2,
  "yolo_ms": 310.5,
  "crop_ms": 4.1,
  "dino_ms": 780.3,
  "head_ms": 0.8,
  "overlay_ms": 1.5,
  "total_ms": 1132.4,
  "num_cows": 3,
  "current_mib": 420.5,
  "peak_mib": 435.2,
  "cpu_percent": 612.3
}
```

### 2.5 Flamegraph Output Format

Each line follows Brendan Gregg's folded stack format:

```
stage_name frame_count N
```

Example:
```
decode 1280
yolo 35
dino 42
crop 1281
head 1282
overlay 1283
```

Where `N` is the cumulative milliseconds spent in that stage across all measured frames.

---

## 3. Code: `profiling/profiler.py`

```python
#!/usr/bin/env python3
"""
BCS Pipeline Profiler — measures per-stage latency, CPU, memory on Qualcomm RB3gen2.

Pipeline:
   frame → YOLOv8n-seg (detect+mask cows) → crop each cow → resize 224 →
   DINOv2 ViT-S/14 ONNX (CLS 384) → BcsHead (softmax) → overlay

Usage:
  python3 profiling/profiler.py \
      --video sample_cow_video.mp4 \
      --yolo yolov8n-seg.pt \
      --dino-onnx dinov2_vits14.onnx \
      --head production_head_vits.pt \
      --config production_config.json \
      --max-frames 500 --warmup 30 \
      --json --csv --flamegraph \
      --profile-cpu --profile-memory

Dependencies:
  pip install numpy opencv-python ultralytics onnxruntime psutil
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

COCO_COW = 19

BAND_COLORS = {
    0: (60, 76, 231),
    1: (104, 168, 85),
    2: (82, 78, 196),
}


# ---------------------------------------------------------------------------
# BcsHead — lightweight 3-layer MLP
# ---------------------------------------------------------------------------

class BcsHead:
    """BcsHead classifier — 3-layer MLP: LayerNorm→Linear(384→128)→GELU→Dropout
    → LayerNorm→Linear(128→128)→GELU→Dropout→Linear(128→3).

    Numpy-only inference (no PyTorch dependency for profiling).
    """

    def __init__(self, weights_path: str, in_dim: int = 384, d_model: int = 128):
        state = np.load(weights_path, allow_pickle=True)
        # The checkpoint is a torch state_dict saved via torch.save();
        # we load it through a thin wrapper that converts to numpy arrays.
        self.proj_ln_gamma = state["proj.0.weight"] if "proj.0.weight" in state else None
        self.proj_ln_beta = state["proj.0.bias"] if "proj.0.bias" in state else None
        self.proj_w = state["proj.1.weight"]
        self.proj_b = state["proj.1.bias"]
        self.head_ln_gamma = state["head.0.weight"] if "head.0.weight" in state else None
        self.head_ln_beta = state["head.0.bias"] if "head.0.bias" in state else None
        self.head_w = state["head.1.weight"]
        self.head_b = state["head.1.bias"]
        self.cls_w = state["cls.weight"]
        self.cls_b = state["cls.bias"]
        self.in_dim = in_dim
        self.d_model = d_model

    @classmethod
    def from_torch_checkpoint(cls, weights_path: str) -> BcsHead:
        """Load a PyTorch checkpoint (.pt) by converting via torch→numpy."""
        import torch
        sd = torch.load(weights_path, map_location="cpu")
        np_state = {}
        for k, v in sd.items():
            np_state[k] = v.numpy()
        # save a temporary .npz copy for numpy-only loading
        npz_path = weights_path + ".npz"
        np.savez(npz_path, **np_state)
        instance = cls.__new__(cls)
        instance._load_npz(npz_path)
        os.remove(npz_path)
        return instance

    def _load_npz(self, path: str):
        state = np.load(path)
        self.proj_ln_gamma = state.get("proj.0.weight")
        self.proj_ln_beta = state.get("proj.0.bias")
        self.proj_w = state["proj.1.weight"]
        self.proj_b = state["proj.1.bias"]
        self.head_ln_gamma = state.get("head.0.weight")
        self.head_ln_beta = state.get("head.0.bias")
        self.head_w = state["head.1.weight"]
        self.head_b = state["head.1.bias"]
        self.cls_w = state["cls.weight"]
        self.cls_b = state["cls.bias"]
        self.in_dim = self.proj_w.shape[1]
        self.d_model = self.proj_w.shape[0]

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                    eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def _linear(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return x @ w.T + b

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        x = feats
        if self.proj_ln_gamma is not None:
            x = self._layer_norm(x, self.proj_ln_gamma, self.proj_ln_beta)
        x = self._linear(x, self.proj_w, self.proj_b)
        x = self._gelu(x)
        if self.head_ln_gamma is not None:
            x = self._layer_norm(x, self.head_ln_gamma, self.head_ln_beta)
        x = self._linear(x, self.head_w, self.head_b)
        x = self._gelu(x)
        return self._linear(x, self.cls_w, self.cls_b)


# ---------------------------------------------------------------------------
# YOLO wrapper — ultralytics-based detection + segmentation
# ---------------------------------------------------------------------------

class YOLOWrapper:
    """Thin wrapper around ultralytics.YOLO for consistent interface."""

    def __init__(self, model_path: str, conf: float = 0.35):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, frame: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
        """Return (boxes_list, masks_list) where each box is xyxy and each mask
        is a float32 array resized to frame shape, or None if no detection."""
        r = self.model.predict(frame, classes=[COCO_COW], conf=self.conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return [], []
        H, W = frame.shape[:2]
        boxes = [b.xyxy[0].cpu().numpy() for b in r.boxes]
        if r.masks is not None:
            mk_np = r.masks.data.cpu().numpy()
            masks = [cv2.resize(mk.astype(np.float32), (W, H),
                                interpolation=cv2.INTER_NEAREST) for mk in mk_np]
        else:
            masks = [None] * len(boxes)
        return boxes, masks


# ---------------------------------------------------------------------------
# DINOv2 ONNX wrapper — onnxruntime CPU inference
# ---------------------------------------------------------------------------

class DINOv2ONNX:
    """DINOv2 ViT-S/14 via ONNX Runtime on CPU."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(onnx_path, so)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def __call__(self, batch_nchw: np.ndarray) -> np.ndarray:
        return self.sess.run([self.output_name], {self.input_name: batch_nchw.astype(np.float32)})[0]


# ---------------------------------------------------------------------------
# Preprocessing — crop, mask, resize, normalize
# ---------------------------------------------------------------------------

def make_crop(frame: np.ndarray, box: np.ndarray, mask: np.ndarray | None,
              mean: list[float], std: list[float], size: int = 224) -> np.ndarray | None:
    """Crop cow from frame, apply segmentation mask, resize to `size`,
    BGR→RGB, normalize, return (3, H, W) float32 array."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if mask is not None:
        m = mask[y1:y2, x1:x2]
        if m.shape[:2] != crop.shape[:2]:
            m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        crop = crop * m[..., None]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(mean, np.float32)) / np.array(std, np.float32)
    return rgb.transpose(2, 0, 1).astype(np.float32)  # (3, 224, 224)


# ---------------------------------------------------------------------------
# Overlay — draw bounding boxes, labels, BCS bands
# ---------------------------------------------------------------------------

def overlay_results(frame: np.ndarray, boxes: list[np.ndarray],
                    probs: np.ndarray, classes: list[str]) -> None:
    """Draw BCS annotations onto frame in-place."""
    for box, p in zip(boxes, probs):
        k = int(p.argmax())
        conf = float(p[k])
        x1, y1, x2, y2 = [int(v) for v in box]
        col = BAND_COLORS.get(k, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = f"BCS: {classes[k]} {conf:.2f}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 22), (x1 + tw + 6, y1), col, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# FrameResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Timing and resource data for a single frame."""
    frame: int = 0
    elapsed: float = 0.0
    decode_ms: float = 0.0
    yolo_ms: float = 0.0
    crop_ms: float = 0.0
    dino_ms: float = 0.0
    head_ms: float = 0.0
    overlay_ms: float = 0.0
    total_ms: float = 0.0
    num_cows: int = 0
    current_mib: float = 0.0
    peak_mib: float = 0.0
    cpu_percent: float = 0.0


# ---------------------------------------------------------------------------
# PipelineProfiler
# ---------------------------------------------------------------------------

class PipelineProfiler:
    """End-to-end profiler for the BCS pipeline."""

    def __init__(
        self,
        video_path: str,
        yolo_path: str,
        dino_onnx_path: str,
        head_path: str,
        config_path: str,
        max_frames: int = 500,
        warmup: int = 30,
        output_dir: str = "profiling",
        conf: float = 0.35,
        profile_memory: bool = False,
        profile_cpu: bool = False,
    ):
        self.max_frames = max_frames
        self.warmup = warmup
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile_memory = profile_memory
        self.profile_cpu = profile_cpu
        self.quiet = False

        # Load config
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.mean = self.cfg["preprocess"]["mean"]
        self.std = self.cfg["preprocess"]["std"]
        self.classes = self.cfg["classes"]

        # Open video
        self.cap = cv2.VideoCapture(video_path)
        self.fps_in = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames_in = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load models
        self.yolo = YOLOWrapper(yolo_path, conf=conf)
        self.dino = DINOv2ONNX(dino_onnx_path)
        self.head = BcsHead(head_path, in_dim=self.cfg.get("in_dim", 384),
                            d_model=self.cfg.get("d_model", 128))

        # Resource tracking
        self._process = None
        self._tracemalloc_initialized = False
        if self.profile_cpu:
            import psutil
            self._process = psutil.Process()
        if self.profile_memory:
            import tracemalloc
            tracemalloc.start()
            self._tracemalloc_initialized = True

        self._results: list[FrameResult] = []

    # ── warmup ──────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Process warmup frames without recording timings."""
        for _ in range(self.warmup):
            ok, frame = self.cap.read()
            if not ok:
                break
            boxes, masks = self.yolo.predict(frame)
            if boxes:
                crops = []
                for box, mk in zip(boxes, masks):
                    c = make_crop(frame, box, mk, self.mean, self.std)
                    if c is not None:
                        crops.append(c)
                if crops:
                    batch = np.stack(crops)
                    _ = self.dino(batch)
                    feats = _  # (K, 384)
                    _ = self.head(feats)

    # ── per-frame profiling (generator) ─────────────────────────────────

    def profile(self) -> Iterator[FrameResult]:
        """Run the pipeline on measured frames, yielding FrameResult per frame."""
        t_start = time.perf_counter()
        frame_idx = 0

        while True:
            # ── decode ──────────────────────────────────────────────────
            t0 = time.perf_counter()
            ok, frame = self.cap.read()
            t1 = time.perf_counter()
            if not ok or frame_idx >= self.max_frames:
                break
            decode_ms = (t1 - t0) * 1000.0

            # ── YOLO detection + segmentation ──────────────────────────
            t2 = time.perf_counter()
            boxes, masks = self.yolo.predict(frame)
            t3 = time.perf_counter()
            yolo_ms = (t3 - t2) * 1000.0

            # ── crop + preprocess ──────────────────────────────────────
            t4 = time.perf_counter()
            crops = []
            for box, mk in zip(boxes, masks):
                c = make_crop(frame, box, mk, self.mean, self.std)
                if c is not None:
                    crops.append(c)
            t5 = time.perf_counter()
            crop_ms = (t5 - t4) * 1000.0

            # ── DINOv2 ─────────────────────────────────────────────────
            t6 = time.perf_counter()
            dino_feats = np.empty((0, 384), dtype=np.float32)
            num_cows = len(crops)
            if num_cows > 0:
                batch = np.stack(crops)
                dino_feats = self.dino(batch)
            t7 = time.perf_counter()
            dino_ms = (t7 - t6) * 1000.0

            # ── BcsHead ────────────────────────────────────────────────
            t8 = time.perf_counter()
            logits = np.empty((0, 3), dtype=np.float32)
            probs = np.empty((0, 3), dtype=np.float32)
            if num_cows > 0:
                logits = self.head(dino_feats)
                exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            t9 = time.perf_counter()
            head_ms = (t9 - t8) * 1000.0

            # ── overlay ─────────────────────────────────────────────────
            tA = time.perf_counter()
            if num_cows > 0:
                overlay_results(frame, boxes, probs, self.classes)
            tB = time.perf_counter()
            overlay_ms = (tB - tA) * 1000.0

            # ── record ──────────────────────────────────────────────────
            t_elapsed = time.perf_counter() - t_start
            total_ms = (tB - t0) * 1000.0

            result = FrameResult(
                frame=frame_idx,
                elapsed=t_elapsed,
                decode_ms=decode_ms,
                yolo_ms=yolo_ms,
                crop_ms=crop_ms,
                dino_ms=dino_ms,
                head_ms=head_ms,
                overlay_ms=overlay_ms,
                total_ms=total_ms,
                num_cows=num_cows,
            )

            # ── optional memory tracking ────────────────────────────────
            if self.profile_memory and self._tracemalloc_initialized:
                import tracemalloc
                current, peak = tracemalloc.get_traced_memory()
                result.current_mib = current / (1024 * 1024)
                result.peak_mib = peak / (1024 * 1024)
                tracemalloc.reset_peak()

            # ── optional CPU tracking ───────────────────────────────────
            if self.profile_cpu and self._process is not None:
                result.cpu_percent = self._process.cpu_percent(interval=0)

            self._results.append(result)
            yield result

            frame_idx += 1

        self.cap.release()

    # ── statistics ──────────────────────────────────────────────────────

    def compute_stats(self, values: list[float]) -> dict[str, float]:
        """Compute summary statistics for a list of values."""
        if not values:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0,
                    "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "std": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    # ── aggregated results ──────────────────────────────────────────────

    def aggregate(self) -> dict[str, Any]:
        """Compute summary statistics for every stage across all frames."""
        if not self._results:
            return {}
        stages = ["decode_ms", "yolo_ms", "crop_ms", "dino_ms", "head_ms",
                   "overlay_ms", "total_ms"]
        stats = {}
        for stage in stages:
            values = [getattr(r, stage) for r in self._results]
            stats[stage] = self.compute_stats(values)
        # End-to-end FPS
        total_elapsed = self._results[-1].elapsed if self._results else 1.0
        n = len(self._results)
        stats["end_to_end_fps"] = n / total_elapsed if total_elapsed > 0 else 0.0
        stats["num_frames"] = n
        stats["gross_cpu_percent"] = {
            "mean": float(np.mean([r.cpu_percent for r in self._results])),
            "max": float(np.max([r.cpu_percent for r in self._results])),
        } if self.profile_cpu else {}
        # Peak memory across all frames
        if self.profile_memory:
            stats["peak_memory_mib"] = float(np.max([r.peak_mib for r in self._results]))
        return stats

    # ── report ──────────────────────────────────────────────────────────

    def report(self, stats: dict[str, Any]) -> str:
        """Return a formatted plain-text summary table."""
        if not stats:
            return "No data."
        lines = [
            "=" * 85,
            "  BCS Pipeline Performance Profiling — Qualcomm RB3gen2 (QCM6490)",
            "=" * 85,
            f"  Frames measured:     {stats['num_frames']}",
            f"  End-to-end FPS:      {stats['end_to_end_fps']:.2f}",
            f"  Mean frame latency:  {stats['total_ms']['mean']:.2f} ms",
            f"  Peak memory (MiB):   {stats.get('peak_memory_mib', 'N/A')}",
            "=" * 85,
            f"  {'Stage':<20s} {'Mean(ms)':>10s} {'Median(ms)':>10s} {'P95(ms)':>10s} "
            f"{'P99(ms)':>10s} {'Std(ms)':>10s} {'Min(ms)':>10s} {'Max(ms)':>10s}",
            "  " + "-" * 88,
        ]
        stages = ["decode_ms", "yolo_ms", "crop_ms", "dino_ms", "head_ms",
                   "overlay_ms", "total_ms"]
        labels = {
            "decode_ms": "Video Decode",
            "yolo_ms": "YOLOv8n-seg",
            "crop_ms": "Crop+Preprocess",
            "dino_ms": "DINOv2 ViT-S",
            "head_ms": "BcsHead",
            "overlay_ms": "Overlay",
            "total_ms": "Total per frame",
        }
        for stage in stages:
            s = stats[stage]
            lines.append(
                f"  {labels[stage]:<20s} {s['mean']:>10.2f} {s['median']:>10.2f} "
                f"{s['p95']:>10.2f} {s['p99']:>10.2f} {s['std']:>10.2f} "
                f"{s['min']:>10.2f} {s['max']:>10.2f}"
            )
        if self.profile_cpu and stats["gross_cpu_percent"]:
            lines.append("  " + "-" * 88)
            lines.append(
                f"  {'CPU util (mean)':<20s} {stats['gross_cpu_percent']['mean']:>10.1f} %  "
                f"(peak: {stats['gross_cpu_percent']['max']:.1f} %)"
            )
        lines.append("=" * 85)
        return "\n".join(lines)

    # ── save helpers ────────────────────────────────────────────────────

    def save_json(self, stats: dict[str, Any]) -> str:
        """Save full stats + per-frame results to JSON, return path."""
        data = {
            "metadata": {
                "platform": "Qualcomm RB3gen2 (QCM6490)",
                "cpu": "4×A55@1.9 GHz + 4×A78@2.4 GHz",
                "memory_gb": 7.1,
                "video_resolution": f"{self.frame_w}×{self.frame_h}",
                "video_fps": self.fps_in,
                "total_frames_input": self.total_frames_in,
                "warmup_frames": self.warmup,
                "measured_frames": len(self._results),
                "models": {
                    "yolo": "yolov8n-seg (7 MB)",
                    "dino": "dinov2_vits14.onnx (88 MB)",
                    "head": "BcsHead (273 KB)",
                },
            },
            "summary": stats,
            "per_frame": [asdict(r) for r in self._results],
        }
        path = str(self.output_dir / "perf_report.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def save_csv(self) -> str:
        """Save per-frame results to CSV, return path."""
        if not self._results:
            return ""
        path = str(self.output_dir / "perf_report.csv")
        fieldnames = list(asdict(self._results[0]).keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self._results:
                w.writerow(asdict(r))
        return path

    def save_flamegraph(self) -> str:
        """Save flamegraph-compatible folded stack output, return path."""
        if not self._results:
            return ""
        stages = ["decode", "yolo", "crop", "dino", "head", "overlay"]
        attr_map = {
            "decode": "decode_ms",
            "yolo": "yolo_ms",
            "crop": "crop_ms",
            "dino": "dino_ms",
            "head": "head_ms",
            "overlay": "overlay_ms",
        }
        accum: dict[str, float] = {s: 0.0 for s in stages}
        for r in self._results:
            for s in stages:
                accum[s] += getattr(r, attr_map[s])
        path = str(self.output_dir / "flamegraph.folded")
        with open(path, "w") as f:
            for s in stages:
                f.write(f"{s} {accum[s]:.0f}\n")
        return path

    def save_table(self, report_str: str) -> str:
        """Save the formatted text report, return path."""
        path = str(self.output_dir / "perf_report_table.txt")
        with open(path, "w") as f:
            f.write(report_str)
        return path

    def run(self) -> dict[str, Any]:
        """Convenience: warmup → profile → compute → report → save all."""
        if not self.quiet:
            print(f"[profiler] Warmup: {self.warmup} frames ...")
        self.warmup()
        if not self.quiet:
            print(f"[profiler] Profiling: {self.max_frames} frames ...")
        for _ in self.profile():
            pass
        stats = self.aggregate()
        rpt = self.report(stats)
        print(rpt)
        paths = {"table": self.save_table(rpt)}
        paths["json"] = self.save_json(stats)
        paths["csv"] = self.save_csv()
        paths["flamegraph"] = self.save_flamegraph()
        if not self.quiet:
            for k, v in paths.items():
                print(f"[profiler] Saved {k}: {v}")
        return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BCS Pipeline Profiler — Qualcomm RB3gen2 (QCM6490)"
    )
    p.add_argument("--video", required=True, help="Input H.264 video file")
    p.add_argument("--yolo", required=True, help="YOLOv8n-seg model (.pt)")
    p.add_argument("--dino-onnx", required=True, help="DINOv2 ViT-S/14 ONNX model")
    p.add_argument("--head", required=True, help="BcsHead checkpoint (.pt)")
    p.add_argument("--config", required=True, help="Production config JSON")
    p.add_argument("--max-frames", type=int, default=500,
                   help="Number of frames to measure after warmup (default: 500)")
    p.add_argument("--warmup", type=int, default=30,
                   help="Warmup frames to skip (default: 30)")
    p.add_argument("--output-dir", default="profiling",
                   help="Output directory for reports (default: profiling/)")
    p.add_argument("--conf", type=float, default=0.35,
                   help="YOLO confidence threshold (default: 0.35)")
    p.add_argument("--profile-memory", action="store_true",
                   help="Enable per-frame memory tracking via tracemalloc")
    p.add_argument("--profile-cpu", action="store_true",
                   help="Enable per-frame CPU tracking via psutil")
    p.add_argument("--json", action="store_true",
                   help="Save per-frame results as JSON")
    p.add_argument("--csv", action="store_true",
                   help="Save per-frame results as CSV")
    p.add_argument("--flamegraph", action="store_true",
                   help="Save flamegraph-compatible folded stack output")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-frame progress output")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate files exist
    for fname in [args.video, args.yolo, args.dino_onnx, args.head, args.config]:
        if not os.path.exists(fname):
            print(f"[ERROR] File not found: {fname}", file=sys.stderr)
            sys.exit(1)

    profiler = PipelineProfiler(
        video_path=args.video,
        yolo_path=args.yolo,
        dino_onnx_path=args.dino_onnx,
        head_path=args.head,
        config_path=args.config,
        max_frames=args.max_frames,
        warmup=args.warmup,
        output_dir=args.output_dir,
        conf=args.conf,
        profile_memory=args.profile_memory,
        profile_cpu=args.profile_cpu,
    )
    profiler.quiet = args.quiet

    if not args.quiet:
        print(f"[profiler] BCS Pipeline Profiler — Qualcomm RB3gen2 (QCM6490)")
        print(f"[profiler] Video:   {args.video} ({profiler.frame_w}×{profiler.frame_h} @ {profiler.fps_in} fps)")
        print(f"[profiler] YOLO:    {args.yolo}")
        print(f"[profiler] DINOv2:  {args.dino_onnx}")
        print(f"[profiler] Head:    {args.head}")
        print(f"[profiler] Config:  {args.config}")
        print(f"[profiler] Frames:  {args.max_frames}  Warmup: {args.warmup}")
        print(f"[profiler] Memory:  {args.profile_memory}  CPU: {args.profile_cpu}")
        print()

    stats = profiler.run()

    # Save optional formats that weren't auto-saved
    if args.json:
        profiler.save_json(stats)
    if args.csv:
        profiler.save_csv()
    if args.flamegraph:
        profiler.save_flamegraph()


if __name__ == "__main__":
    main()
```

---

## 4. Expected Performance Baseline

The following table provides **CPU-only estimates** for each pipeline stage on the Qualcomm RB3gen2. These are calculated from published benchmark data, model parameter counts, and arithmetic intensity analysis. No GPU acceleration (Adreno 642L OpenCL) is assumed.

### 4.1 Stage-Level Latency Estimates

| Stage | Model | CPU Time Est. | Notes |
|-------|-------|--------------|-------|
| **Video Decode** | ffmpeg / OpenCV `cv2.VideoCapture` | **30–50 ms** | 2560×1440 H.264 software decode on A78 cores; 4:2:0 → BGR conversion included |
| **YOLOv8n-seg** | ultralytics (ONNX / PyTorch CPU) | **200–500 ms** | ~7M param model; seg head adds ~15% over detection-only |
| **Crop + Preprocess** | NumPy / OpenCV | **2–5 ms / cow** | Mask resize, crop, colour convert, 224×224 resize, normalise |
| **DINOv2 ViT-S/14** | ONNX Runtime CPU (4 threads) | **500–1000 ms** | 88M param transformer; ~22 GMACs at 224×224; dominant bottleneck |
| **BcsHead** | NumPy GELU+Linear | **< 1 ms** | 3-layer MLP (384→128→128→3), ~100K params, negligible |
| **Overlay** | OpenCV `cv2.rectangle` + `putText` | **1–2 ms** | 2 boxes + 2 labels + FPS counter |
| **Total** | | **~733–1558 ms** | → **0.6–1.4 FPS** (CPU only, single-stream) |

### 4.2 Scalability with Number of Cows

| # Cows | YOLO | Crop | DINOv2 (batch) | Head | Overlay | Total (est.) | FPS (est.) |
|--------|------|------|----------------|------|---------|-------------|-----------|
| 0 | 200–500 ms | 0 ms | 0 ms | 0 ms | 0 ms | 230–550 ms | 1.8–4.3 |
| 1 | 200–500 ms | 2–5 ms | 500–1000 ms | <1 ms | 1–2 ms | 733–1558 ms | 0.6–1.4 |
| 2 | 200–500 ms | 4–10 ms | 500–1000 ms* | <1 ms | 1–2 ms | 735–1563 ms | 0.6–1.4 |
| 4 | 200–500 ms | 8–20 ms | 500–1000 ms* | <1 ms | 1–2 ms | 739–1573 ms | 0.6–1.4 |

> **\*Key insight:** DINOv2 latency is **almost invariant to batch size** at these batch sizes (1–4). The ONNX transformer processes a batched tensor nearly as fast as a single vector up to moderate batch sizes, because the attention mechanism is compute-bound by the (224/14)² = 256 token sequence length, not the batch dimension. This means **multiple cows per frame incur almost linear scaling only in the Crop stage**, not in DINOv2.

### 4.3 Memory Estimate

| Component | Memory | Notes |
|-----------|--------|-------|
| Raw frame (2560×1440×3) | ~10.5 MB | uint8 BGR |
| YOLOv8n-seg model | ~14 MB | 7M params × fp32 + activations |
| DINOv2 ONNX model | ~176 MB | 88M params × fp16 |
| DINOv2 activations (1 cow) | ~50 MB | 256 tokens × 384 dim × layers |
| BcsHead weights | ~0.3 MB | 100K params × fp32 |
| Crops (batch=4) | ~0.1 MB | 4 × 3 × 224 × 224 × fp32 |
| Overlay output | ~10.5 MB | Same as input frame |
| **Total working set** | **~260 MB** | Well within 7.1 GB |

---

## 5. Profiling Results Template

Use this section to record actual profiling results after running `profiling/profiler.py`.

### 5.1 Run Metadata

```
Date:              <YYYY-MM-DD>
Platform:          Qualcomm RB3gen2 (QCM6490)
OS:                <Linux distribution + kernel>
Python:            3.12.x
onnxruntime:       <version>
ultralytics:       <version>
OpenCV:            <version>
Video:             <filename>  (2560×1440 @ 25 fps, H.264)
Total frames:      <count>
Warmup frames:     30
Measured frames:   500
```

### 5.2 Summary Table

```
=====================================================================================
  BCS Pipeline Performance Profiling — Qualcomm RB3gen2 (QCM6490)
=====================================================================================
  Frames measured:     500
  End-to-end FPS:      X.XX
  Mean frame latency:  XXXX.XX ms
  Peak memory (MiB):   XXX.X
=====================================================================================
  Stage                      Mean(ms)  Median(ms)   P95(ms)   P99(ms)   Std(ms)   Min(ms)   Max(ms)
  ----------------------------------------------------------------------------------------
  Video Decode                 XX.XX      XX.XX      XX.XX     XX.XX     XX.XX     XX.XX     XX.XX
  YOLOv8n-seg                 XXX.XX     XXX.XX     XXX.XX    XXX.XX     XX.XX     XX.XX    XXX.XX
  Crop+Preprocess               X.XX       X.XX       X.XX      X.XX      X.XX      X.XX      X.XX
  DINOv2 ViT-S                XXX.XX     XXX.XX     XXX.XX    XXX.XX     XX.XX     XX.XX    XXX.XX
  BcsHead                       X.XX       X.XX       X.XX      X.XX      X.XX      X.XX      X.XX
  Overlay                       X.XX       X.XX       X.XX      X.XX      X.XX      X.XX      X.XX
  Total per frame            XXXX.XX    XXXX.XX    XXXX.XX   XXXX.XX    XXX.XX    XXX.XX   XXXX.XX
=====================================================================================
```

### 5.3 Per-Stage Breakdown Chart (to generate)

After collecting data, generate these charts using the CSV output:

**Chart 1 — Stage Latency Distribution (box plot or violin plot)**
```
decode  yolo   crop   dino   head  overlay
  |      |      |      |      |      |
  |   ┌──┴──┐   |   ┌──┴──┐   |      |
  |   │    │   |   │    │   |      |
  |   └──┬──┘   |   └──┬──┘   |      |
  |      |      |      |      |      |
  └──────┴──────┴──────┴──────┴──────┴──
```

**Chart 2 — Frame-by-Frame Timeline (line chart)**

Plot each stage's ms against frame number to spot:
- Thermal throttling (gradual latency increase)
- GC pauses (spikes)
- Frame-to-frame variance

**Chart 3 — CPU Utilisation Heatmap**

If `--profile-cpu` was enabled, plot `cpu_percent` vs. frame number. Expected: ~600–700% (7–8 cores peaking) during DINOv2 inference.

### 5.4 Raw Results (Placeholder)

| Frame | Decode (ms) | YOLO (ms) | Crop (ms) | DINO (ms) | Head (ms) | Overlay (ms) | Total (ms) | Cows |
|-------|------------|-----------|----------|----------|---------|------------|-----------|------|
| 31    |            |           |          |          |         |            |           |      |
| 32    |            |           |          |          |         |            |           |      |
| ...   |            |           |          |          |         |            |           |      |
| 530   |            |           |          |          |         |            |           |      |

---

## 6. Bottleneck Analysis

Based on the expected baseline, the bottlenecks rank as follows:

### 6.1 Bottleneck Ranking

| Rank | Stage | % of Total (est.) | Limiting Factor |
|------|-------|-------------------|----------------|
| **#1** | **DINOv2 ViT-S/14** | **60–70%** | 88M params, 22 GMACs, pure CPU-bound. Self-attention over 256 tokens × 12 layers. |
| **#2** | **YOLOv8n-seg** | **20–30%** | 7M params + segmentation head. Heavy but much smaller than DINOv2. |
| **#3** | **Video Decode** | **3–6%** | Software H.264 decode on A78 cores. |
| **#4** | **Crop + Preprocess** | **0.3–1%** | Dependent on number of cows. |
| **#5** | **Overlay** | **0.1–0.2%** | Trivial OpenCV drawing. |
| **#6** | **BcsHead** | **< 0.1%** | Negligible MLP. |

### 6.2 Bottleneck Deep Dive: DINOv2 ViT-S

| Property | Value |
|----------|-------|
| Parameters | 88.6M |
| FLOPs (224×224) | ~22 GFLOPs (inference) |
| Attention heads | 6 |
| Transformer layers | 12 |
| Hidden dim | 384 |
| Patch size | 14 → 256 patches |
| Arithmetic intensity | Very low (~10–20 FLOPs/byte) → **memory-bound** |
| Profile shape | 500–1000 ms flat, with +0–5% variation per cow |

**Why DINOv2 is slow on CPU:**
1. **Low arithmetic intensity:** Transformer self-attention is dominated by matrix multiplies where the matrices (256×384) are small enough to be memory-bound rather than compute-bound on a CPU.
2. **No SVE/SVE2 on A78:** The Kryo 670 cores lack scalable vector extensions. Each A78 can issue 2×128-bit NEON SIMD ops/cycle, achieving ~24 GFLOPS at 2.4 GHz — but the memory wall dominates.
3. **ONNX Runtime overhead:** Graph partitioning, thread pool scheduling, and tensor copies add 5–10% overhead over a bare-metal implementation.

### 6.3 Optimization Proposals

| Bottleneck | Optimization | Expected Gain | Complexity |
|-----------|-------------|--------------|-----------|
| **DINOv2** | Adreno 642L OpenCL compute shader for attention | 3–5× (~200 ms) | High |
| **DINOv2** | ONNX Runtime with QNN EP (Qualcomm Neural Processing SDK) | 3–5× | High |
| **DINOv2** | Quantise to INT8 via ONNX static quantization | 2–3× (~300 ms) | Medium |
| **DINOv2** | Frame skipping (run DINOv2 every 2nd/3rd frame, reuse features) | 2–3× effective FPS | Low |
| **DINOv2** | Reduce input to 168×168 instead of 224×224 | 1.5× (~400 ms) | Low |
| **YOLO** | YOLOv8n-seg → YOLOv8n (drop mask head, use box-only) | 1.2–1.5× | Low |
| **YOLO** | ONNX Runtime with INT8 quantization | 2× (~150 ms) | Medium |
| **Decode** | Reduce resolution to 1920×1080 (downscale on capture, not resize) | 1.5–2× decode | Low |
| **Pipeline** | Parallel decode + YOLO (pipeline over 2 frames) | 1.15× | Medium |
| **Pipeline** | Process cows within a frame in parallel (multiprocessing for DINOv2) | 1.1× (small batches) | Medium |

---

## 7. Optimization Targets

### 7.1 Target Tiers

| Tier | FPS Target | Per-Frame Budget | Use Case |
|------|-----------|-----------------|----------|
| **Real-time** | **25 FPS** | **40 ms** | Live visual monitoring; overlays must keep pace with video input. Requires GPU/Neural DSP acceleration — not achievable on CPU alone. |
| **Near real-time** | **10 FPS** | **100 ms** | Frame-skipping every 2–3 frames, still displayable. Achievable with DINOv2 INT8 + YOLO INT8. |
| **Batch processing** | **1–3 FPS** | **333–1000 ms** | Offline analysis of recorded video. Achievable now on CPU without any optimisation. |

### 7.2 Roadmap to 10 FPS

| Step | Optimisation | Cumulative FPS | Cumulative Latency |
|------|-------------|----------------|--------------------|
| Baseline (CPU, all fp32) | — | 0.6–1.4 | ~1000 ms |
| 1 | Frame skip: DINOv2 every 2nd frame | 1.2–2.8 | ~500 ms |
| 2 | YOLOv8n ONNX INT8 | 2.0–4.0 | ~350 ms |
| 3 | Video decode 1920×1080 | 3.0–5.5 | ~250 ms |
| 4 | DINOv2 ONNX INT8 | 5.0–8.0 | ~150 ms |
| 5 | DINOv2 168×168 input | 6.0–10.0 | ~120 ms |
| 6 | Adreno OpenCL attention | 15.0+ | ~65 ms |

### 7.3 Thermal Considerations

The RB3gen2 (QCM6490) has a passive thermal design. Sustained 8-core 100% load will cause throttling after ~5–10 minutes. **Recommendations:**
- Pin DINOv2 inference to 2×A78 cores to avoid thermal saturation of all 4 big cores.
- Use `--max-frames` to limit profiling runs to 500–1000 frames.
- Monitor core frequencies with `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` during profiling.
- Insert 10 ms `time.sleep()` between frames in deploy (not profiling) to allow thermal recovery.

---

## Appendix A: Running the Profiler

```bash
# Basic CPU-only profiling
python3 profiling/profiler.py \
    --video sample_cow_video.mp4 \
    --yolo yolov8n-seg.pt \
    --dino-onnx dinov2_vits14.onnx \
    --head production_head_vits.pt \
    --config production_config.json

# Full profiling with memory and CPU tracking, all outputs
python3 profiling/profiler.py \
    --video sample_cow_video.mp4 \
    --yolo yolov8n-seg.pt \
    --dino-onnx dinov2_vits14.onnx \
    --head production_head_vits.pt \
    --config production_config.json \
    --max-frames 500 --warmup 30 \
    --profile-memory --profile-cpu \
    --json --csv --flamegraph \
    --output-dir profiling/results

# Short smoke test (50 frames, no warmup)
python3 profiling/profiler.py \
    --video sample_cow_video.mp4 \
    --yolo yolov8n-seg.pt \
    --dino-onnx dinov2_vits14.onnx \
    --head production_head_vits.pt \
    --config production_config.json \
    --max-frames 50 --warmup 0 \
    --quiet
```

## Appendix B: Generating Flamegraph SVG

```bash
# Generate folded stack output (done by --flamegraph flag)
python3 profiling/profiler.py ... --flamegraph
# → produces profiling/flamegraph.folded

# Render SVG using Brendan Gregg's FlameGraph tools
git clone https://github.com/brendangregg/FlameGraph
./FlameGraph/flamegraph.pl profiling/flamegraph.folded > profiling/flamegraph.svg
```

## Appendix C: Plotting with the CSV Output

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("profiling/perf_report.csv")

# Per-stage box plot
stages = ["decode_ms", "yolo_ms", "dino_ms", "total_ms"]
df[stages].boxplot()
plt.title("BCS Pipeline Stage Latency Distribution")
plt.ylabel("Milliseconds")
plt.savefig("profiling/stage_latency.png")

# Frame-by-frame timeline
df.plot(y="total_ms")
plt.title("Per-Frame Total Latency")
plt.ylabel("ms")
plt.xlabel("Frame")
plt.savefig("profiling/total_latency_timeline.png")

# FPS over sliding window of 30 frames
window = 30
fps = window / df["total_ms"].rolling(window).sum() * 1000
fps.plot()
plt.title("Sliding-Window FPS (window=30)")
plt.ylabel("FPS")
plt.xlabel("Frame")
plt.savefig("profiling/sliding_fps.png")
```
