#!/usr/bin/env python3
"""
Unified Master Edge Pipeline Runner
Automatically detects underlying hardware (NVIDIA Jetson, Qualcomm RB3 Gen2, Radxa CM5 RK3588, or CPU)
and dispatches execution to the corresponding zero-copy pipeline architecture.

Usage:
  python run_all_platforms.py --video sample_cow_video.mp4
  python run_all_platforms.py --target jetson --video sample_cow_video.mp4
  python run_all_platforms.py --target qualcomm --video sample_cow_video.mp4
  python run_all_platforms.py --target radxa --video sample_cow_video.mp4
"""

import sys
import os
import platform
import argparse
import subprocess

def detect_hardware_target() -> str:
    """Detects host platform silicon architecture."""
    # Check for NVIDIA Jetson Orin
    if os.path.exists("/etc/nv_tegra_release") or os.path.exists("/usr/local/cuda"):
        return "jetson"
    # Check for Rockchip RK3588
    elif os.path.exists("/proc/device-tree/compatible"):
        try:
            with open("/proc/device-tree/compatible", "r") as f:
                content = f.read()
                if "rk3588" in content or "radxa" in content:
                    return "radxa"
                elif "qcom" in content or "qcm6490" in content:
                    return "qualcomm"
        except Exception:
            pass
    # Check for Qualcomm environment variables or QNN SDK
    if os.environ.get("QAIRT_SDK_ROOT") or os.environ.get("QNN_SDK_ROOT"):
        return "qualcomm"

    # Default fallback
    return "cpu"


def run_pipeline(target: str, video_path: str, config_path: str):
    print("=================================================")
    print("      Unified Master Edge BCS Pipeline          ")
    print("=================================================")
    print(f"Detected/Selected Target: [{target.upper()}]")
    print(f"Video Source            : {video_path}")
    print(f"Configuration File      : {config_path}")

    if target == "jetson":
        print("[INFO] Launching NVIDIA Jetson Orin NVMM Pipeline...")
        script = os.path.join("jetson_orin_nano", "scripts", "run_jetson.py")
        cmd = [sys.executable, script, "--video", video_path, "--config", config_path]
    elif target == "radxa":
        print("[INFO] Launching Radxa CM5 (RK3588) MPP/RGA/RKNN Pipeline...")
        script = os.path.join("radxa_cm5", "scripts", "run_rk3588.py")
        cmd = [sys.executable, script, "--video", video_path, "--config", config_path]
    elif target == "qualcomm":
        print("[INFO] Launching Qualcomm RB3 Gen2 ION DMA-BUF Pipeline...")
        cmd = [sys.executable, "-m", "qualcomm_adaptation", "--video", video_path, "--config", config_path]
    else:
        print("[INFO] Launching CPU Fallback ONNX Runtime Pipeline...")
        cmd = [sys.executable, "-m", "qualcomm_adaptation", "--video", video_path, "--config", config_path, "--benchmark"]

    print(f"[CMD] Executing: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Master Edge BCS Runner")
    parser.add_argument("--target", choices=["auto", "jetson", "qualcomm", "radxa", "cpu"], default="auto", help="Hardware target platform")
    parser.add_argument("--video", default="sample_cow_video.mp4", help="Video stream file path")
    parser.add_argument("--config", default="production_config.json", help="Configuration file path")
    args = parser.parse_args()

    selected_target = args.target if args.target != "auto" else detect_hardware_target()
    run_pipeline(selected_target, args.video, args.config)
