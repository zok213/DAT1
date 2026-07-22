#!/usr/bin/env python3
"""
Direct Local-to-Colab T4 Execution Script
Directly triggers or executes model compilation on remote Google Colab T4 GPU from local CLI.

Modes:
  1. SSH Direct Mode: Executes remote command directly over SSH tunnel.
  2. Watchdog Trigger Mode: Drops a trigger file into synced Google Drive folder.

Usage:
  python scripts/direct_colab_runner.py --mode ssh --host xxx.trycloudflare.com --command "python scripts/compile_all_models.py"
  python scripts/direct_colab_runner.py --mode watchdog --drive-dir "C:/Users/.../Google Drive/DAT1"
"""

import os
import sys
import argparse
import subprocess
import json
import time

def run_ssh_direct(host: str, port: int, command: str):
    print("=================================================")
    print("      Direct SSH Local-to-Colab Execution       ")
    print("=================================================")
    print(f"Remote Colab Host : {host}:{port}")
    print(f"Executing Command : {command}")

    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-p", str(port),
        f"root@{host}",
        f"cd /content/DAT1 && {command}"
    ]

    print(f"[CMD] {' '.join(ssh_cmd)}")
    subprocess.run(ssh_cmd)


def run_watchdog_trigger(drive_dir: str):
    print("=================================================")
    print("   Google Drive Local-to-Colab Watchdog Trigger  ")
    print("=================================================")
    print(f"Google Drive Path: {drive_dir}")

    trigger_file = os.path.join(drive_dir, "TRIGGER_COMPILE.json")
    payload = {
        "timestamp": time.time(),
        "action": "compile_all",
        "output_dir": "models/"
    }

    with open(trigger_file, "w") as f:
        json.dump(payload, f)

    print(f"[OK] Dropped trigger payload -> {trigger_file}")
    print("[INFO] Google Colab T4 watchdog will detect and execute compilation within 5 seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct Local-to-Colab T4 Executor")
    parser.add_argument("--mode", choices=["ssh", "watchdog"], default="ssh", help="Execution mode")
    parser.add_argument("--host", default="", help="Cloudflare SSH host (e.g. xxx.trycloudflare.com)")
    parser.add_argument("--port", type=int, default=22, help="SSH Port")
    parser.add_argument("--command", default="python scripts/compile_all_models.py --output-dir models/ --quantize int8", help="Remote command to execute")
    parser.add_argument("--drive-dir", default="", help="Local path to Google Drive synced folder")
    args = parser.parse_args()

    if args.mode == "ssh":
        if not args.host:
            print("[ERROR] Please provide --host (e.g. --host xxx.trycloudflare.com)")
            sys.exit(1)
        run_ssh_direct(args.host, args.port, args.command)
    elif args.mode == "watchdog":
        if not args.drive-dir:
            print("[ERROR] Please provide --drive-dir path")
            sys.exit(1)
        run_watchdog_trigger(args.drive_dir)
