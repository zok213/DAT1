#!/usr/bin/env python3
"""
BCS Benchmark Suite — Qualcomm RB3gen2
=======================================

Runs structured benchmarks across multiple configurations to profile
performance. Generates JSON output + optional visualization.

Usage:
  python3 scripts/benchmark.py \
      --video sample_cow_video.mp4 \
      --yolo yolov8n-seg.pt \
      --dino-onnx dinov2_vits14.onnx \
      --head production_head_vits.pt \
      --config production_config.json \
      --output profiling/benchmark_results.json

  # Run all scaling benchmarks
  python3 scripts/benchmark.py [paths] --all-scaling

  # Quick test (100 frames)
  python3 scripts/benchmark.py [paths] --max-frames 100
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


BENCH_SCENARIOS = {
    "cpu_baseline": {
        "label": "CPU Baseline (no opts)",
        "frame_skip": 0,
        "input_scale": 1.0,
        "conf": 0.35,
        "description": "Full resolution, every frame — worst case",
    },
    "skip_2": {
        "label": "Frame Skip 2",
        "frame_skip": 2,
        "input_scale": 1.0,
        "conf": 0.35,
        "description": "Process every 3rd frame",
    },
    "scale_05": {
        "label": "Half Resolution",
        "frame_skip": 0,
        "input_scale": 0.5,
        "conf": 0.35,
        "description": "1280×720 input",
    },
    "skip_2_scale_05": {
        "label": "Skip 2 + Half Res",
        "frame_skip": 2,
        "input_scale": 0.5,
        "conf": 0.35,
        "description": "Combined optimization",
    },
    "skip_5_scale_05": {
        "label": "Skip 5 + Half Res",
        "frame_skip": 5,
        "input_scale": 0.5,
        "conf": 0.35,
        "description": "Aggressive — ~10+ FPS expected",
    },
    "high_conf": {
        "label": "Higher Confidence (0.5)",
        "frame_skip": 0,
        "input_scale": 1.0,
        "conf": 0.50,
        "description": "Fewer detections, faster",
    },
}


def run_benchmark(video: str, yolo: str, dino_onnx: str, head: str,
                  config: str, scenario: str, params: dict,
                  max_frames: int = 500, warmup: int = 30,
                  hw_decode: bool = False) -> dict:
    """Run a single benchmark scenario and return timing results."""
    cmd = [
        sys.executable, "-m", "qualcomm_adaptation",
        "--video", video,
        "--yolo", yolo,
        "--dino-onnx", dino_onnx,
        "--head", head,
        "--config", config,
        "--benchmark",                    # no overlay drawing
        "--profile",
        f"--frame-skip={params['frame_skip']}",
        f"--input-scale={params['input_scale']}",
        f"--conf={params['conf']}",
        f"--max-frames={max_frames + warmup}",
    ]
    if hw_decode:
        cmd.append("--hw-decode")

    print(f"\n{'='*60}")
    print(f"Benchmark: {params['label']}")
    print(f"{'='*60}")
    print(f"  Config: {scenario}")
    print(f"  Cmd:    {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    # Parse stdout for perf data
    stdout = result.stdout
    stderr = result.stderr

    # Extract performance lines
    perf: dict = {
        "scenario": scenario,
        "label": params["label"],
        "description": params["description"],
        "params": params,
        "elapsed_clock": round(elapsed, 2),
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr[:2000] if stderr else "",
    }

    # Parse metrics from output
    for line in stdout.split("\n"):
        if "[perf]" in line:
            parts = line.replace("[perf]", "").strip().split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                value = parts[1]
                try:
                    perf[key.lower().replace("/", "_per_")] = float(value)
                except ValueError:
                    perf[key.lower().replace("/", "_per_")] = value

        if "[done]" in line:
            if "frames" in line:
                try:
                    perf["frames_processed"] = int(line.split()[1])
                except (IndexError, ValueError):
                    pass

    if result.returncode != 0:
        print(f"  [FAIL] Return code {result.returncode}")
        print(f"  STDERR: {stderr[:500]}")
    else:
        fps = perf.get("fps", perf.get("end2end", "?"))
        print(f"  [OK] FPS={fps}, time={elapsed:.1f}s")

    return perf


def main():
    ap = argparse.ArgumentParser(description="BCS Benchmark Suite")
    ap.add_argument("--video", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--dino-onnx", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="profiling/benchmark_results.json")
    ap.add_argument("--max-frames", type=int, default=500,
                    help="Frames to process per scenario")
    ap.add_argument("--warmup", type=int, default=30,
                    help="Warmup frames (not counted)")
    ap.add_argument("--scenarios", nargs="+",
                    default=list(BENCH_SCENARIOS.keys()),
                    choices=list(BENCH_SCENARIOS.keys()) + ["all"],
                    help="Which scenarios to run")
    ap.add_argument("--all-scaling", action="store_true",
                    help="Run all RESOLUTION × SKIP combinations")
    ap.add_argument("--hw-decode", action="store_true",
                    help="Use GStreamer V4L2 HW-accelerated video decode")
    args = ap.parse_args()

    # Validate files
    for f in [args.video, args.yolo, args.dino_onnx, args.head, args.config]:
        if not os.path.isfile(f):
            print(f"[ERR] File not found: {f}")
            sys.exit(1)

    # Build scenario list
    if args.all_scaling:
        scenarios = {}
        for skip in [0, 1, 2, 5]:
            for scale in [1.0, 0.75, 0.5, 0.33]:
                name = f"skip{skip}_scale{scale}"
                scenarios[name] = {
                    "label": f"Skip {skip}, Scale {scale:.2f}",
                    "frame_skip": skip,
                    "input_scale": scale,
                    "conf": 0.35,
                    "description": f"frame_skip={skip}, input_scale={scale}",
                }
    elif args.scenarios and args.scenarios[0] == "all":
        scenarios = BENCH_SCENARIOS
    else:
        scenarios = {k: BENCH_SCENARIOS[k] for k in args.scenarios
                     if k in BENCH_SCENARIOS}

    print(f"\nBCS Benchmark Suite — {len(scenarios)} scenarios")
    print(f"  Video:    {args.video}")
    print(f"  Frames:   {args.max_frames} per scenario (+{args.warmup} warmup)")
    print(f"  Output:   {args.output}")

    results: list[dict] = []
    for name, params in scenarios.items():
        perf = run_benchmark(
            video=args.video, yolo=args.yolo,
            dino_onnx=args.dino_onnx, head=args.head,
            config=args.config, scenario=name,
            params=params, max_frames=args.max_frames,
            warmup=args.warmup, hw_decode=args.hw_decode,
        )
        results.append(perf)

        # Save after each scenario (in case of crash)
        _save_intermediate(args.output, results)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scenario':<25} {'FPS':>8} {'Frames':>8} {'Time':>8}")
    print("-" * 60)
    for r in results:
        fps = r.get("end2end", r.get("fps", "N/A"))
        frames = r.get("frames_processed", "?")
        elapsed = r.get("elapsed_clock", "?")
        print(f"{r['label']:<25} {str(fps):>8} {str(frames):>8} {str(elapsed):>8}s")

    # Final save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "video": args.video,
                "max_frames": args.max_frames,
                "warmup": args.warmup,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "scenarios": results,
        }, f, indent=2)
    print(f"\n[ok] Results saved → {output_path}")


def _save_intermediate(path: str, results: list):
    """Save intermediate results to prevent data loss."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"scenarios": results}, f, indent=2)


if __name__ == "__main__":
    main()
