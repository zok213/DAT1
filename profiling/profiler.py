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
from dataclasses import dataclass, asdict
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
        state = np.load(weights_path, allow_pickle=True).item() if weights_path.endswith('.npz') else \
                torch_load_to_dict(weights_path)
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
        self.in_dim = in_dim
        self.d_model = d_model

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


def torch_load_to_dict(path: str) -> dict[str, np.ndarray]:
    """Load a PyTorch .pt checkpoint and convert all tensors to numpy arrays."""
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.numpy() for k, v in sd.items()}


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
# DINOv2 TFLite wrapper — XNNPACK CPU inference
# ---------------------------------------------------------------------------

class DINOv2TFLite:
    """DINOv2 ViT-S/14 via TFLite (ai_edge_litert) on CPU."""

    def __init__(self, tflite_path: str):
        from ai_edge_litert.interpreter import Interpreter
        self.interpreter = Interpreter(model_path=tflite_path, num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    def __call__(self, batch_nchw: np.ndarray) -> np.ndarray:
        # Resize input tensor to match batch size
        self.interpreter.resize_tensor_input(self.input_details['index'], batch_nchw.shape)
        self.interpreter.allocate_tensors()
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details['index'], batch_nchw.astype(np.float32))
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        return self.interpreter.get_tensor(self.output_details['index'])


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
    return rgb.transpose(2, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Overlay — draw bounding boxes, labels, BCS bands
# ---------------------------------------------------------------------------

def overlay_results(frame: np.ndarray, boxes: list[np.ndarray],
                    probs: np.ndarray, classes: list[str]) -> None:
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
        self.warmup_frames = warmup
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile_memory = profile_memory
        self.profile_cpu = profile_cpu
        self.quiet = False

        with open(config_path) as f:
            self.cfg = json.load(f)
        self.mean = self.cfg["preprocess"]["mean"]
        self.std = self.cfg["preprocess"]["std"]
        self.classes = self.cfg["classes"]

        self.cap = cv2.VideoCapture(video_path)
        self.fps_in = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames_in = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.yolo = YOLOWrapper(yolo_path, conf=conf)
        if dino_onnx_path.endswith('.tflite'):
            self.dino = DINOv2TFLite(dino_onnx_path)
        else:
            self.dino = DINOv2ONNX(dino_onnx_path)
        self.head = BcsHead(head_path, in_dim=self.cfg.get("in_dim", 384),
                            d_model=self.cfg.get("d_model", 128))

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

    def warmup(self) -> None:
        for _ in range(self.warmup_frames):
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
                    feats = self.dino(batch)
                    self.head(feats)

    def profile(self) -> Iterator[FrameResult]:
        t_start = time.perf_counter()
        frame_idx = 0

        while True:
            t0 = time.perf_counter()
            ok, frame = self.cap.read()
            t1 = time.perf_counter()
            if not ok or frame_idx >= self.max_frames:
                break
            decode_ms = (t1 - t0) * 1000.0

            t2 = time.perf_counter()
            boxes, masks = self.yolo.predict(frame)
            t3 = time.perf_counter()
            yolo_ms = (t3 - t2) * 1000.0

            t4 = time.perf_counter()
            crops = []
            for box, mk in zip(boxes, masks):
                c = make_crop(frame, box, mk, self.mean, self.std)
                if c is not None:
                    crops.append(c)
            t5 = time.perf_counter()
            crop_ms = (t5 - t4) * 1000.0

            t6 = time.perf_counter()
            dino_feats = np.empty((0, 384), dtype=np.float32)
            num_cows = len(crops)
            if num_cows > 0:
                batch = np.stack(crops)
                dino_feats = self.dino(batch)
            t7 = time.perf_counter()
            dino_ms = (t7 - t6) * 1000.0

            t8 = time.perf_counter()
            logits = np.empty((0, 3), dtype=np.float32)
            probs = np.empty((0, 3), dtype=np.float32)
            if num_cows > 0:
                logits = self.head(dino_feats)
                exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            t9 = time.perf_counter()
            head_ms = (t9 - t8) * 1000.0

            tA = time.perf_counter()
            if num_cows > 0:
                overlay_results(frame, boxes, probs, self.classes)
            tB = time.perf_counter()
            overlay_ms = (tB - tA) * 1000.0

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

            if self.profile_memory and self._tracemalloc_initialized:
                import tracemalloc
                current, peak = tracemalloc.get_traced_memory()
                result.current_mib = current / (1024 * 1024)
                result.peak_mib = peak / (1024 * 1024)
                tracemalloc.reset_peak()

            if self.profile_cpu and self._process is not None:
                result.cpu_percent = self._process.cpu_percent(interval=0)

            self._results.append(result)
            yield result

            frame_idx += 1

        self.cap.release()

    def compute_stats(self, values: list[float]) -> dict[str, float]:
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

    def aggregate(self) -> dict[str, Any]:
        if not self._results:
            return {}
        stages = ["decode_ms", "yolo_ms", "crop_ms", "dino_ms", "head_ms",
                   "overlay_ms", "total_ms"]
        stats = {}
        for stage in stages:
            values = [getattr(r, stage) for r in self._results]
            stats[stage] = self.compute_stats(values)
        total_elapsed = self._results[-1].elapsed if self._results else 1.0
        n = len(self._results)
        stats["end_to_end_fps"] = n / total_elapsed if total_elapsed > 0 else 0.0
        stats["num_frames"] = n
        stats["gross_cpu_percent"] = {
            "mean": float(np.mean([r.cpu_percent for r in self._results])),
            "max": float(np.max([r.cpu_percent for r in self._results])),
        } if self.profile_cpu else {}
        if self.profile_memory:
            stats["peak_memory_mib"] = float(np.max([r.peak_mib for r in self._results]))
        return stats

    def report(self, stats: dict[str, Any]) -> str:
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

    def save_json(self, stats: dict[str, Any]) -> str:
        data = {
            "metadata": {
                "platform": "Qualcomm RB3gen2 (QCM6490)",
                "cpu": "4×A55@1.9 GHz + 4×A78@2.4 GHz",
                "memory_gb": 7.1,
                "video_resolution": f"{self.frame_w}×{self.frame_h}",
                "video_fps": self.fps_in,
                "total_frames_input": self.total_frames_in,
                "warmup_frames": self.warmup_frames,
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
        path = str(self.output_dir / "perf_report_table.txt")
        with open(path, "w") as f:
            f.write(report_str)
        return path

    def run(self) -> dict[str, Any]:
        if not self.quiet:
            print(f"[profiler] Warmup: {self.warmup_frames} frames ...")
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

    if args.json:
        profiler.save_json(stats)
    if args.csv:
        profiler.save_csv()
    if args.flamegraph:
        profiler.save_flamegraph()


if __name__ == "__main__":
    main()
