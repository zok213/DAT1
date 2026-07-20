#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Environment Setup — BCS Pipeline on Qualcomm RB3gen2
# ═══════════════════════════════════════════════════════════════════════════════
# Usage: bash scripts/setup_environment.sh [--venv-only] [--full]
#
# --venv-only : Create venv and install pip packages only
# --full      : System packages + venv + model conversions
# (default)   : System packages + venv
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

log()  { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERR]${NC}  $1"; }

# ── 1. System dependencies ──────────────────────────────────────────────────
install_system_deps() {
    log "Installing system dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3-pip \
        python3-opencv \
        python3-dev \
        libjpeg-dev \
        zlib1g-dev \
        libgl1 \
        ninja-build \
        cmake \
        build-essential \
        gstreamer1.0-tools \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        2>&1 | tail -5
    log "System dependencies installed"
}

# ── 2. Python virtual environment ────────────────────────────────────────────
setup_venv() {
    log "Creating Python virtual environment..."
    python3 -m venv venv --system-site-packages
    source venv/bin/activate

    log "Installing Python packages..."
    pip install --upgrade pip setuptools wheel 2>&1 | tail -2

    # Core ML / CV packages
    pip install numpy==1.26.4 2>&1 | tail -2
    pip install opencv-python-headless 2>&1 | tail -2
    pip install onnx onnxruntime 2>&1 | tail -2
    pip install ultralytics 2>&1 | tail -2

    # Optional: PyTorch for aarch64 (CPU only / piwheels)
    log "Attempting PyTorch install for aarch64..."
    pip install torch --index-url https://piwheels.org/simple 2>&1 | tail -5 || \
        warn "PyTorch from piwheels failed. Install manually from https://github.com/Kashu7100/pytorch-arm64"

    # Profiling extras
    pip install psutil matplotlib 2>&1 | tail -2

    log "Packages installed:"
    pip list 2>/dev/null | grep -iE "torch|onnx|numpy|opencv|ultralytics"

    deactivate
}

# ── 3. Model conversions ─────────────────────────────────────────────────────
convert_models() {
    source venv/bin/activate

    log "Converting BcsHead to ONNX..."
    python3 scripts/convert_head_to_onnx.py \
        --input production_head_vits.pt \
        --output models/bcs_head.onnx \
        --config production_config.json

    log "Exporting YOLOv8n-seg to ONNX..."
    python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt')
model.export(format='onnx', imgsz=224, half=False)
import os, shutil
if os.path.exists('yolov8n-seg.onnx'):
    shutil.move('yolov8n-seg.onnx', 'models/yolov8n-seg.onnx')
    print('YOLO ONNX exported to models/yolov8n-seg.onnx')
" 2>&1 | tail -5

    log "DINOv2 is already in ONNX format at dinov2_vits14.onnx"
    log "Copying to models/ for organization..."
    cp dinov2_vits14.onnx models/ 2>/dev/null || true

    deactivate
    log "Model conversions complete"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║   BCS Pipeline — Qualcomm RB3gen2 Environment Setup        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    local mode="${1:-default}"

    case "$mode" in
        --venv-only)
            setup_venv
            ;;
        --full)
            install_system_deps
            setup_venv
            convert_models
            ;;
        *)
            install_system_deps
            setup_venv
            ;;
    esac

    echo ""
    log "Setup complete!"
    echo ""
    echo "  Activate:  source venv/bin/activate"
    echo "  Run demo:  python3 -m qualcomm_adaptation --video sample_cow_video.mp4 \\"
    echo "                 --yolo yolov8n-seg.pt --dino-onnx dinov2_vits14.onnx \\"
    echo "                 --head production_head_vits.pt --config production_config.json"
    echo "  Benchmark: python3 -m qualcomm_adaptation --video sample_cow_video.mp4 \\"
    echo "                 --yolo yolov8n-seg.pt --dino-onnx dinov2_vits14.onnx \\"
    echo "                 --head production_head_vits.pt --config production_config.json \\"
    echo "                 --benchmark --profile"
    echo ""
}

main "$@"
