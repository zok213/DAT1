#!/usr/bin/env python3
"""
BCS Runner — Qualcomm RB3gen2 Adapted
======================================

Usage:
  # Tier 1:  CPU-only with PyTorch BcsHead + Ultralytics + ONNX DINOv2
  python3 -m qualcomm_adaptation --video sample_cow_video.mp4 \
      --yolo yolov8n-seg.pt --dino-onnx dinov2_vits14.onnx \
      --head production_head_vits.pt --config production_config.json

  # Tier 1b: Pure NumPy BcsHead (no PyTorch at all)
  python3 -m qualcomm_adaptation --video sample_cow_video.mp4 \
      --yolo yolov8n-seg.pt --dino-onnx dinov2_vits14.onnx \
      --head production_head_vits.pt --config production_config.json \
      --head-backend numpy

  # Tier 2: All ONNX Runtime (export YOLO to ONNX first)
  python3 -m qualcomm_adaptation --video sample_cow_video.mp4 \
      --yolo-onnx yolov8n-seg.onnx --dino-onnx dinov2_vits14.onnx \
      --head-onnx bcs_head.onnx --config production_config.json

  # With optimizations:
  python3 -m qualcomm_adaptation --video sample_cow_video.mp4 \
      --yolo yolov8n-seg.pt --dino-onnx dinov2_vits14.onnx \
      --head production_head_vits.pt --config production_config.json \
      --frame-skip 2 --input-scale 0.5 --benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .config import BCSConfig
from .pipeline import (
    YoloUltralytics,
    DinoONNX,
    DinoQNN,
    BcsHeadNumPy,
    BcsHeadONNX,
    BcsHeadTorch,
    VideoReaderHW,
    make_crop,
    draw_overlay,
)


def setup_argparse() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="BCS Inference — Qualcomm RB3gen2 Adapted",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Input / Output
    ap.add_argument("--video", required=True, help="Input video file path")
    ap.add_argument("--out", default="bcs_annotated.mp4", help="Output video path")

    # Model backends
    ap.add_argument("--yolo", default=None, help="YOLO .pt path (ultralytics)")
    ap.add_argument("--yolo-onnx", default=None, help="YOLO .onnx path (raw ONNX)")
    ap.add_argument("--dino-onnx", default="dinov2_vits14.onnx", help="DINOv2 .onnx path")
    ap.add_argument("--head", default="production_head_vits.pt", help="BcsHead weights")
    ap.add_argument("--head-onnx", default=None, help="BcsHead ONNX path")
    ap.add_argument("--config", default="production_config.json", help="Config JSON")

    # Backend selection
    ap.add_argument("--head-backend", choices=["torch", "numpy", "onnx"], default="torch",
                    help="BcsHead backend (default: torch)")
    ap.add_argument("--dino-backend", choices=["onnx", "qnn"], default="onnx",
                    help="DINOv2 backend: onnx (ONNX Runtime, default) or qnn (QAIRT)")
    ap.add_argument("--dino-qnn-binary", default=None,
                    help="QNN context binary (.bin) for DINOv2 QNN backend")
    ap.add_argument("--dino-qnn-backend", choices=["CPU", "HTP", "GPU"], default="CPU",
                    help="QNN backend type for DINOv2 (default: CPU)")

    # Tuning
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--frame-skip", type=int, default=0,
                    help="Process every (N+1)th frame; 0 = all")
    ap.add_argument("--input-scale", type=float, default=1.0,
                    help="Downscale input (0.5 = half res)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--num-threads", type=int, default=0,
                    help="ONNX Runtime intra_op_num_threads (0 = auto for big.LITTLE)")

    # Modes
    ap.add_argument("--benchmark", action="store_true",
                    help="Benchmark mode: no overlay, print per-frame timing JSON")
    ap.add_argument("--display", action="store_true", help="Show cv2 window")
    ap.add_argument("--no-ood-warning", action="store_true",
                    help="Skip the 'OOD: screening only' overlay")
    ap.add_argument("--hw-decode", action="store_true",
                    help="Use GStreamer V4L2 HW-accelerated video decode (Qualcomm msm_vidc)")

    # Profiling
    ap.add_argument("--profile", action="store_true",
                    help="Enable detailed per-frame profiling output")
    return ap


def main():
    args = setup_argparse().parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = BCSConfig.from_json(
        args.config,
        yolo_conf=args.conf,
        frame_skip=args.frame_skip,
        input_scale=args.input_scale,
        max_frames=args.max_frames,
        benchmark=args.benchmark,
        no_display=not args.display,
        hw_decode=args.hw_decode,
        num_threads=args.num_threads,
    )
    print(f"[init] Config: {len(cfg.classes)} classes, resize={cfg.resize}, "
          f"frame_skip={cfg.frame_skip}, input_scale={cfg.input_scale}")

    # ── Validate models exist ────────────────────────────────────────────────
    for name, p in [("dino-onnx", args.dino_onnx), ("head", args.head),
                    ("config", args.config)]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{name} not found: {p}")

    # ── Init YOLO ────────────────────────────────────────────────────────────
    yolo = None
    if args.yolo:
        print(f"[init] YOLO: ultralytics backend ({args.yolo})")
        yolo = YoloUltralytics(args.yolo, conf=args.conf)
    elif args.yolo_onnx:
        print(f"[init] YOLO: ONNX backend ({args.yolo_onnx})")
        # yolo = YoloONNX(args.yolo_onnx, conf=args.conf)  # requires full NMS
        raise NotImplementedError("ONNX YOLO backend needs complete NMS. Use --yolo .pt for now.")
    else:
        print("[init] No YOLO model specified; using center-crop scoring (--no-detect mode)")

    # ── Init DINOv2 ──────────────────────────────────────────────────────────
    if args.dino_backend == "qnn":
        qnn_binary = args.dino_qnn_binary
        if not qnn_binary:
            qnn_binary = os.path.join(
                os.path.dirname(os.path.abspath(args.dino_onnx or ".")),
                "dinov2_fp32_cpu.bin.bin"
            ) if args.dino_onnx else None
        if not qnn_binary or not os.path.isfile(qnn_binary):
            raise FileNotFoundError(
                f"DINOv2 QNN binary not found: {qnn_binary}. "
                f"Pass --dino-qnn-binary or ensure the .bin exists next to --dino-onnx."
            )
        print(f"[init] DINOv2: QNN backend ({qnn_binary}), "
              f"qnn_backend={args.dino_qnn_backend}")
        dino = DinoQNN(
            binary_path=qnn_binary,
            backend=args.dino_qnn_backend,
        )
    else:
        print(f"[init] DINOv2: ONNX Runtime ({args.dino_onnx}), "
              f"threads={cfg.num_threads if cfg.num_threads > 0 else 'auto'}")
        dino = DinoONNX(args.dino_onnx, providers=["CPUExecutionProvider"],
                        num_threads=cfg.num_threads)

    # ── Init BcsHead ─────────────────────────────────────────────────────────
    head_backend = args.head_backend
    if args.head_onnx:
        head_backend = "onnx"

    if head_backend == "onnx":
        head_path = args.head_onnx or args.head.replace(".pt", ".onnx")
        print(f"[init] BcsHead: ONNX backend ({head_path})")
        head = BcsHeadONNX(head_path, providers=["CPUExecutionProvider"])
    elif head_backend == "numpy":
        print(f"[init] BcsHead: NumPy backend ({args.head})")
        head = BcsHeadNumPy(args.head, in_dim=cfg.dino_in_dim, d=cfg.head_d_model)
    else:
        print(f"[init] BcsHead: PyTorch backend ({args.head})")
        head = BcsHeadTorch(args.head, device="cpu")

    # ── Open video ───────────────────────────────────────────────────────────
    if args.hw_decode:
        print(f"[init] Video: GStreamer V4L2 HW decode ({args.video})")
        cap = VideoReaderHW(args.video)
        orig_fps = cap.fps
        orig_w = cap.width
        orig_h = cap.height
        print(f"        HW decode active: {cap.using_hw}")
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {args.video}")
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Apply input scale
    if cfg.input_scale < 1.0:
        W = int(orig_w * cfg.input_scale)
        H = int(orig_h * cfg.input_scale)
    else:
        W, H = orig_w, orig_h

    print(f"[init] Video: {orig_w}×{orig_h} @ {orig_fps:.1f} FPS → "
          f"{W}×{H} (scale={cfg.input_scale})")

    # ── Video writer ─────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, orig_fps / (cfg.frame_skip + 1),
                             (W, H) if not args.benchmark else (1, 1))

    # ── Main loop ────────────────────────────────────────────────────────────
    n_processed = 0
    n_total = 0
    t_yolo = 0.0
    t_crop = 0.0
    t_dino = 0.0
    t_head = 0.0
    t_overlay = 0.0
    t_total = 0.0
    profiling_data: list[dict] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Frame skipping
        if cfg.frame_skip > 0 and n_total % (cfg.frame_skip + 1) != 0:
            n_total += 1
            continue

        # Scale frame if needed
        if cfg.input_scale < 1.0:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

        t0 = time.perf_counter()
        crops, boxes = [], []

        # ── YOLO ─────────────────────────────────────────────────────────
        t1 = t0
        if yolo is not None:
            t1a = time.perf_counter()
            detected_boxes, detected_masks = yolo(frame)
            t1 = time.perf_counter()
            t_yolo += t1 - t1a

            if detected_boxes is not None:
                masks_list = (detected_masks if detected_masks is not None
                              else [None] * len(detected_boxes))
                for b, mk in zip(detected_boxes, masks_list):
                    mask_for_crop = mk if isinstance(mk, np.ndarray) else None
                    c = make_crop(frame, b, mask_for_crop, cfg.mean, cfg.std, cfg.resize)
                    if c is not None:
                        crops.append(c)
                        boxes.append(b)
        else:
            # No YOLO: center-crop
            s = min(H, W)
            cx, cy = W // 2, H // 2
            box = np.array([cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2],
                           dtype=np.float32)
            c = make_crop(frame, box, None, cfg.mean, cfg.std, cfg.resize)
            if c is not None:
                crops.append(c)
                boxes.append(box)

        # ── DINOv2 ───────────────────────────────────────────────────────
        t2 = time.perf_counter()
        if crops:
            batch = np.stack(crops).astype(np.float32)  # (K, 3, 224, 224)
            feats = dino(batch)                         # (K, 384)
        else:
            feats = np.empty((0, cfg.dino_in_dim), dtype=np.float32)
        t3 = time.perf_counter()
        t_crop += t2 - t1
        t_dino += t3 - t2

        # ── BcsHead ──────────────────────────────────────────────────────
        if len(feats) > 0:
            logits = head(feats)
            probs = softmax(logits)
        else:
            probs = np.empty((0, cfg.n_classes))
        t4 = time.perf_counter()
        t_head += t4 - t3

        # ── Overlay ──────────────────────────────────────────────────────
        if not args.benchmark and len(boxes) > 0:
            frame = draw_overlay(
                frame, np.array(boxes), probs, cfg.classes,
                n_processed / max(t_total, 1e-6),
                ood_warning=not args.no_ood_warning,
            )

        t5 = time.perf_counter()
        t_overlay += t5 - t4
        t_total += t5 - t0
        n_processed += 1
        n_total += 1

        # ── Profiling ────────────────────────────────────────────────────
        if args.profile:
            profiling_data.append({
                "frame": n_total,
                "n_cows": len(crops),
                "t_yolo_ms": 1000 * (t1 - t1a) if yolo else 0,
                "t_crop_ms": 1000 * (t2 - t1),
                "t_dino_ms": 1000 * (t3 - t2),
                "t_head_ms": 1000 * (t4 - t3),
                "t_overlay_ms": 1000 * (t5 - t4),
                "t_total_ms": 1000 * (t5 - t0),
            })

        # ── Write / display ──────────────────────────────────────────────
        if not args.benchmark:
            writer.write(frame)
        if args.display:
            cv2.imshow("BCS (Qualcomm)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.max_frames and n_processed >= args.max_frames:
            break

        # Progress
        if n_processed % 50 == 0:
            print(f"  [{n_processed}] {n_processed / max(t_total, 1e-6):5.1f} FPS | "
                  f"{len(crops)} cows in frame")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # ── Report ────────────────────────────────────────────────────────────────
    if n_processed == 0:
        print("[done] No frames processed")
        return

    avg_fps = n_processed / t_total
    avg_yolo = 1000 * t_yolo / n_processed
    avg_crop = 1000 * t_crop / n_processed
    avg_dino = 1000 * t_dino / n_processed
    avg_head = 1000 * t_head / n_processed
    avg_overlay = 1000 * t_overlay / n_processed

    print("-" * 60)
    print(f"[done] {n_processed} frames → {args.out}")
    print(f"[perf] end2end:    {avg_fps:7.1f} FPS ({1000/avg_fps:7.1f} ms/frame)" if avg_fps > 0 else "")
    total_accounted = t_yolo + t_crop + t_dino + t_head + t_overlay
    print(f"[perf] YOLO:       {avg_yolo:7.1f} ms/frame ({100*t_yolo/t_total:.0f}%)" if avg_yolo > 0 else "")
    print(f"[perf] Crop+Prepr: {avg_crop:7.1f} ms/frame")
    print(f"[perf] DINOv2:     {avg_dino:7.1f} ms/frame ({100*t_dino/t_total:.0f}%)")
    print(f"[perf] BcsHead:    {avg_head:7.1f} ms/frame")
    print(f"[perf] Overlay:    {avg_overlay:7.1f} ms/frame")
    print(f"[perf] Total:      {1000/avg_fps:7.1f} ms/frame ({100*total_accounted/t_total:.0f}% accounted)")

    # Save profiling data
    if args.profile:
        out_path = f"profiling_data_{Path(args.video).stem}.json"
        json.dump(profiling_data, open(out_path, "w"), indent=2)
        print(f"[prof] Saved {len(profiling_data)} records → {out_path}")


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    exp = np.exp(x - x_max)
    return exp / exp.sum(axis=axis, keepdims=True)


if __name__ == "__main__":
    main()
