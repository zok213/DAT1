# 🔬 Technical Research Report: Google Colab T4 GPU Disconnection Rules, Limits & Mitigation

> **Document ID:** `reports/14_colab_t4_gpu_disconnection_and_limits_research.md`  
> **Author:** Senior Cloud Infrastructure Architect & AI Systems Specialist  
> **Date:** July 22, 2026  
> **Target:** Google Colab Free Tier Tesla T4 GPU Session Management & Disconnect Recovery  

---

## 📌 Executive Summary of Colab Free T4 GPU Limits

Google Colab provides access to **NVIDIA Tesla T4 GPUs (16 GB VRAM)** free of charge, subject to specific resource limits, idle timers, and session duration caps.

```
                                Google Colab Free T4 GPU Limits
  ┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
  │   90-Minute Idle Cap     │   │   12-Hour Absolute Limit │   │  Daily Usage Quota Cap   │
  │ Disconnects if no stdout │   │ Session recycled at 12h  │   │ ~12 hours T4 GPU / 24h   │
  └──────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘
```

---

## ⏱️ Detailed Disconnection Rules & Triggers

| Limit Type | Duration / Rule | Trigger Condition | Mitigation Strategy |
|---|---|---|---|
| **Idle Timeout** | **90 Minutes** (5,400 seconds) | No user input, terminal command, or output log for 90 min | Run [`scripts/colab_anti_disconnect.py`](file:///d:/Gitrepo/DAT1/scripts/colab_anti_disconnect.py) heartbeat daemon |
| **Max Session Duration** | **12 Hours** (43,200 seconds) | VM automatically reset by Google Cloud container manager | Periodic checkpointing to Google Drive & auto-resume |
| **Browser Tab Close** | **15 - 30 Minutes** | Closing browser tab without active CLI / terminal process | Use `google-colab-cli` background job or keep tab active |
| **GPU Usage Quota** | **~12 Hours / 24 Hours** | Exceeding daily compute allocation ("GPU limit reached") | Automatic fallback to CPU mode or alternate account |
| **RAM Out-Of-Memory** | Instant Termination | RAM exceeds 12.7 GB host RAM threshold | Stream video processing frame-by-frame (never load full video in RAM) |

---

## 🛠️ Automated Mitigation & Auto-Reconnect Tools

### 1. Heartbeat Anti-Disconnect Daemon ([`scripts/colab_anti_disconnect.py`](file:///d:/Gitrepo/DAT1/scripts/colab_anti_disconnect.py))
Executes inside the Colab session, touching stdout every 60 seconds to prevent the 90-minute idle disconnect threshold:

```bash
# Launch Heartbeat Daemon in Background inside Colab
python scripts/colab_anti_disconnect.py &
```

### 2. CLI Auto-Reconnect & Retry Manager ([`scripts/colab_cli_auto_reconnect.py`](file:///d:/Gitrepo/DAT1/scripts/colab_cli_auto_reconnect.py))
Wraps `google-colab-cli` (`colab`) to automatically catch connection drop errors, re-provision a new T4 GPU VM (`colab new --gpu T4`), and resume model building:

```bash
# Run CLI Auto-Reconnect Manager with 3 Retry Attempts
python scripts/colab_cli_auto_reconnect.py --gpu T4 --max-retries 3
```

---

## 💡 Best Practices for Uninterrupted Model Building

1. **Save Checkpoints Frequently**: Always output intermediate model weights (`.onnx`, `.tflite`, `.engine`) to mounted Google Drive (`/content/drive/MyDrive/`) so work is never lost if a 12-hour session ends.
2. **Stream Video Processing**: Process video input frame-by-frame using OpenCV generators to keep RAM RSS under **< 500 MiB**, avoiding OOM crashes.
3. **Monitor Compute Quotas**: If Colab shows *"You cannot connect to a GPU right now"*, use the local ONNX/CPU runner ([`run_all_platforms.py`](file:///d:/Gitrepo/DAT1/run_all_platforms.py)) until your 24-hour quota resets.
