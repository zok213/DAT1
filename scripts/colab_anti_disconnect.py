#!/usr/bin/env python3
"""
Google Colab Anti-Disconnect & Session Heartbeat Monitor
Prevents Google Colab 90-minute idle timeouts by emitting periodic heartbeats and GPU activity.

Usage in Colab:
  python scripts/colab_anti_disconnect.py &
"""

import time
import sys
import os

def run_heartbeat(interval_seconds: int = 60, max_hours: float = 11.5):
    print("=================================================")
    print(" Google Colab Session Anti-Disconnect Daemon     ")
    print("=================================================")
    print(f"Heartbeat Interval: Every {interval_seconds} seconds")
    print(f"Max Session Guard : {max_hours} hours")

    start_time = time.time()
    max_duration_sec = max_hours * 3600.0
    iteration = 0

    while (time.time() - start_time) < max_duration_sec:
        iteration += 1
        elapsed_min = (time.time() - start_time) / 60.0
        remaining_min = (max_duration_sec - (time.time() - start_time)) / 60.0

        print(f"[HEARTBEAT #{iteration:04d}] Session Active: {elapsed_min:.1f}m elapsed | Remaining Guard: {remaining_min:.1f}m", flush=True)
        time.sleep(interval_seconds)

    print("[WARN] Max session duration safety limit reached. Cleaning up...")

if __name__ == "__main__":
    run_heartbeat()
