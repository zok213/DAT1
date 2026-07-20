#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Qualcomm QAIRT (QNN) SDK Setup — BCS Pipeline Acceleration
# ═══════════════════════════════════════════════════════════════════════════════
# Installs the Qualcomm AI Runtime SDK (formerly QNN / Neural Processing SDK)
# for hardware-accelerated inference on RB3gen2's Hexagon CDSP and Adreno GPU.
#
# ✅ Direct download — NO Qualcomm account needed!
#    URL: https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.48.0.260626/v2.48.0.260626.zip
#
# Usage:
#   # Auto-download + install (requires ~85GB free, 2.1GB download)
#   bash scripts/setup_qnn_sdk.sh --auto-download
#
#   # Or with pre-downloaded zip:
#   bash scripts/setup_qnn_sdk.sh --sdk-path /path/to/v2.48.0.260626.zip
#
#   # Model conversion only (SDK already installed):
#   bash scripts/setup_qnn_sdk.sh --convert-only
#
# Architecture Notes:
#   - On ARM64 (RB3gen2): Only ONNX model conversion is supported (no TF/PyTorch)
#   - ONNX → QNN conversion works natively on-device
#   - DLC = Deep Learning Container (QNN model format)
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[QNN]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERR]${NC}  $1"; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

# Latest known working direct download URL
# Updated quarterly by Qualcomm. Check for newer versions at:
# https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community
# or from Neural Processing SDK page: https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai
QNN_DOWNLOAD_URL="${QNN_DOWNLOAD_URL:-https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.48.0.260626/v2.48.0.260626.zip}"
QNN_ZIP_NAME="qnn_sdk_v2.48.0.260626.zip"
QNN_VERSION="2.48.0.260626"

# ── Prerequisites ──────────────────────────────────────────────────────────────
check_prereqs() {
    log "Checking prerequisites..."

    # CDSP must be accessible
    if [ -e /dev/fastrpc-cdsp ]; then
        log "✅ CDSP accessible at /dev/fastrpc-cdsp"
    else
        warn "⚠️  CDSP not accessible at /dev/fastrpc-cdsp"
        warn "   QNN HTP (Hexagon) backend will NOT work"
        warn "   GPU/CPU backends may still work"
    fi

    # Python
    if command -v python3 &>/dev/null; then
        PY_VER=$(python3 --version 2>&1)
        log "✅ $PY_VER"
    else
        err "Python3 not found"
        exit 1
    fi

    # Ubuntu 24.04 check
    if grep -q "24.04" /etc/os-release 2>/dev/null; then
        log "✅ Ubuntu 24.04 detected (supported by SDK)"
    else
        warn "⚠️  Non-24.04 Ubuntu — SDK may not be fully tested on this version"
    fi

    # Disk space (need ~5GB for download + extraction)
    local avail_kb
    avail_kb=$(df "$PROJECT_DIR" --output=avail 2>/dev/null | tail -1)
    if [ "$avail_kb" -gt 5000000 ]; then
        log "✅ Sufficient disk space ($((avail_kb / 1024))MB available)"
    else
        err "Insufficient disk space (need >5GB, have $((avail_kb / 1024))MB)"
        exit 1
    fi

    # Install system dependencies
    log "Installing system dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        make cmake g++ python3-dev libssl-dev \
        libtinfo5 libncurses5 2>&1 | tail -3
    log "✅ Dependencies installed"
}

# ── Download QNN SDK ───────────────────────────────────────────────────────────
download_qnn_sdk() {
    local dest_dir="$1"
    local dest_file="$dest_dir/$QNN_ZIP_NAME"

    if [ -f "$dest_file" ] && [ -s "$dest_file" ]; then
        local size_mb
        size_mb=$(du -m "$dest_file" | cut -f1)
        if [ "$size_mb" -gt 1000 ]; then
            log "✅ SDK already downloaded ($size_mb MB) — skipping download"
            echo "$dest_file"
            return
        fi
    fi

    log "Downloading QNN SDK v$QNN_VERSION..."
    info "URL: $QNN_DOWNLOAD_URL"
    info "Size: ~2.1GB — this will take ~30-60 min"
    echo ""

    mkdir -p "$dest_dir"
    wget --no-check-certificate \
        -O "$dest_file" \
        "$QNN_DOWNLOAD_URL" \
        2>&1 | while IFS= read -r line; do
            # Show progress lines that have percentage
            if [[ "$line" =~ [0-9]+% ]]; then
                echo -ne "\r  Download: $line"
            fi
        done

    echo ""
    if [ -f "$dest_file" ]; then
        local final_size
        final_size=$(du -h "$dest_file" | cut -f1)
        log "✅ Download complete: $final_size"
    else
        err "Download failed"
        exit 1
    fi

    echo "$dest_file"
}

# ── Extract QNN SDK ────────────────────────────────────────────────────────────
install_qnn_sdk() {
    local sdk_zip="$1"
    local install_dir="${2:-$PROJECT_DIR/qnn_sdk}"

    log "Extracting QNN SDK..."
    mkdir -p "$install_dir"

    local extract_start
    extract_start=$(date +%s)
    unzip -o "$sdk_zip" -d "$install_dir" 2>&1 | tail -5
    local extract_end
    extract_end=$(date +%s)

    # Find the QAIRT root (new SDK format) or QNN root (old format)
    QAIRT_ROOT=""
    if [ -d "$install_dir/qairt/$QNN_VERSION" ]; then
        QAIRT_ROOT="$install_dir/qairt/$QNN_VERSION"
    elif [ -d "$install_dir/qairt" ]; then
        QAIRT_ROOT=$(find "$install_dir/qairt" -maxdepth 1 -type d | sort -r | head -1)
    elif [ -d "$install_dir" ]; then
        QAIRT_ROOT=$(find "$install_dir" -maxdepth 2 -name "envsetup.sh" -type f | head -1 | xargs dirname)
    fi

    if [ -z "$QAIRT_ROOT" ] || [ ! -f "$QAIRT_ROOT/bin/envsetup.sh" ]; then
        warn "Could not find QAIRT root automatically"
        warn "Looking in: $install_dir"
        find "$install_dir" -name "envsetup.sh" -type f 2>/dev/null
        # Let the user set it manually
        QAIRT_ROOT="$install_dir/qairt/$QNN_VERSION"
        if [ ! -d "$QAIRT_ROOT" ]; then
            warn "envsetup.sh not found — will need manual setup"
        fi
    else
        log "✅ QNN SDK extracted to: $QAIRT_ROOT"
    fi

    local elapsed=$((extract_end - extract_start))
    local size_mb
    size_mb=$(du -sh "$install_dir" 2>/dev/null | cut -f1)
    log "Extraction: ${elapsed}s | SDK size: $size_mb"

    # Create convenience env file
    cat > "$PROJECT_DIR/qnn_env.sh" << EOF
# QNN SDK Environment — source this file before using QNN tools
#   source qnn_env.sh
export QAIRT_ROOT="$QAIRT_ROOT"
export QNN_SDK_ROOT="\$QAIRT_ROOT"
export SNPE_ROOT="\$QAIRT_ROOT"
export PATH="\$QAIRT_ROOT/bin/aarch64-linux-clang:\$PATH"
export LD_LIBRARY_PATH="\$QAIRT_ROOT/lib/aarch64-linux-clang:\$LD_LIBRARY_PATH"
export PYTHONPATH="\$QAIRT_ROOT/lib/python:\$PYTHONPATH"

# Source the official envsetup if available
if [ -f "\$QAIRT_ROOT/bin/envsetup.sh" ]; then
    source "\$QAIRT_ROOT/bin/envsetup.sh"
fi

echo "QNN SDK ready: \$QAIRT_ROOT"
EOF
    chmod +x "$PROJECT_DIR/qnn_env.sh"
    log "✅ Environment file: source qnn_env.sh"
}

# ── Install SDK dependencies ──────────────────────────────────────────────────
install_sdk_deps() {
    log "Installing QNN SDK Python dependencies..."

    source "$PROJECT_DIR/venv/bin/activate"

    # QNN SDK requires specific versions of ML frameworks
    # On ARM64 (RB3gen2), only ONNX conversion is supported
    pip install --quiet "onnx>=1.12,<1.16" "onnxruntime>=1.17,<2.0" "onnxsim>=0.4" \
        2>&1 | tail -2

    log "✅ SDK Python dependencies installed"

    if [ -f "$QAIRT_ROOT/bin/check-python-dependency" ]; then
        log "Running SDK Python dependency check..."
        sudo bash "$QAIRT_ROOT/bin/check-python-dependency" 2>&1 | tail -5 || true
    fi

    if [ -f "$QAIRT_ROOT/bin/check-linux-dependency.sh" ]; then
        log "Running SDK Linux dependency check..."
        sudo bash "$QAIRT_ROOT/bin/check-linux-dependency.sh" 2>&1 | tail -5 || true
    fi
}

# ── Convert models to QNN DLC format ──────────────────────────────────────────
convert_models() {
    log "Converting models to QNN DLC format..."

    source "$PROJECT_DIR/qnn_env.sh" 2>/dev/null || true
    source "$PROJECT_DIR/venv/bin/activate"

    local qnn_bin="$QAIRT_ROOT/bin/aarch64-linux-clang"
    local converter="$qnn_bin/qnn-onnx-converter"

    if [ ! -f "$converter" ]; then
        # Try x86 version (for cross-compilation)
        converter="$QAIRT_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter"
    fi

    if [ ! -f "$converter" ] && command -v qnn-onnx-converter &>/dev/null; then
        converter="qnn-onnx-converter"
    fi

    if ! command -v "$converter" &>/dev/null; then
        err "qnn-onnx-converter not found!"
        info "Looked in: $converter"
        info "Searching for converter..."
        find "$QAIRT_ROOT" -name "qnn-onnx-converter*" -type f 2>/dev/null | head -5
        return 1
    fi

    log "Using converter: $converter"
    mkdir -p models/qnn

    # 1. DINOv2 ViT-S/14 → QNN
    if [ -f "dinov2_vits14.onnx" ]; then
        log "Converting DINOv2 ViT-S/14 (84MB ONNX → DLC)..."

        # Create input list for conversion
        echo "image 1 3 224 224" > /tmp/dino_inputs.txt

        # FP16 conversion
        log "  → FP16 DLC..."
        python3 "$converter" \
            --input-dlc models/qnn/dinov2_vits14_fp16.dlc \
            --input-network dinov2_vits14.onnx \
            --input-list /tmp/dino_inputs.txt \
            --precision 16 \
            --act_bw 16 2>&1 | tail -3

        if [ -f "models/qnn/dinov2_vits14_fp16.dlc" ]; then
            local size
            size=$(du -h "models/qnn/dinov2_vits14_fp16.dlc" | cut -f1)
            log "  ✅ DINOv2 FP16 DLC: $size"
        fi

        # INT8 quantization (if calibration data exists)
        if [ -d "calibration_data/dino" ]; then
            log "  → INT8 DLC (with calibration)..."
            python3 "$converter" \
                --input-dlc models/qnn/dinov2_vits14_int8.dlc \
                --input-network dinov2_vits14.onnx \
                --input-list calibration_data/dino/inputs.txt \
                --precision 8 \
                --act_bw 8 2>&1 | tail -3

            if [ -f "models/qnn/dinov2_vits14_int8.dlc" ]; then
                local size_i8
                size_i8=$(du -h "models/qnn/dinov2_vits14_int8.dlc" | cut -f1)
                log "  ✅ DINOv2 INT8 DLC: $size_i8 (was 88MB ONNX)"
            fi
        else
            warn "  ⚠️  No calibration data for INT8. Creating dummy calibration..."
            log "  Generating calibration data from random inputs..."
            mkdir -p calibration_data/dino
            for i in $(seq 1 20); do
                python3 -c "
import numpy as np
import cv2
# Create random 224x224 image
img = np.random.randn(3, 224, 224).astype(np.float32)
# Save as raw binary (CHW format)
img.tofile(f'calibration_data/dino/input_$i.raw')
" 2>/dev/null
            done
            echo "calibration_data/dino/input_{1..20}.raw 0" > calibration_data/dino/inputs.txt

            log "  → INT8 DLC (random calibration)..."
            python3 "$converter" \
                --input-dlc models/qnn/dinov2_vits14_int8.dlc \
                --input-network dinov2_vits14.onnx \
                --input-list calibration_data/dino/inputs.txt \
                --precision 8 \
                --act_bw 8 2>&1 | tail -3 || warn "  ⚠️  INT8 conversion failed (needs representative calibration)"
        fi
    fi

    # 2. BcsHead → QNN (tiny model, FP16 is fine)
    if [ -f "models/bcs_head.onnx" ]; then
        log "Converting BcsHead (3KB ONNX → DLC)..."
        echo "input 1 384" > /tmp/bcshead_inputs.txt

        python3 "$converter" \
            --input-dlc models/qnn/bcs_head_fp16.dlc \
            --input-network models/bcs_head.onnx \
            --input-list /tmp/bcshead_inputs.txt \
            --precision 16 \
            --act_bw 16 2>&1 | tail -3

        if [ -f "models/qnn/bcs_head_fp16.dlc" ]; then
            log "  ✅ BcsHead FP16 DLC"
        fi
    fi

    log "✅ Model conversion complete"
    ls -lh models/qnn/ 2>/dev/null
}

# ── Benchmark QNN acceleration ────────────────────────────────────────────────
run_benchmark() {
    log "Running QNN vs CPU benchmark..."

    source "$PROJECT_DIR/qnn_env.sh" 2>/dev/null || true
    source "$PROJECT_DIR/venv/bin/activate"

    # Test QNN backend availability
    log "Testing QNN backends..."
    echo ""

    # Check which backends are available
    local backends=""
    for be in libQnnHtp.so libQnnGpu.so libQnnCpu.so; do
        local be_path
        be_path=$(find "$QAIRT_ROOT" -name "$be" -type f 2>/dev/null | head -1)
        if [ -n "$be_path" ]; then
            local be_name
            be_name=$(basename "$be" .so | sed 's/libQnn//')
            backends="$backends $be_name"
            log "  ✅ $be_name backend: $be_path"
        fi
    done

    if [ -z "$backends" ]; then
        warn "  ⚠️  No QNN backends found for this platform"
        warn "  Expected: HTP (CDSP), GPU (Adreno), CPU"
        return
    fi

    # Run QNN benchmark via qnn-net-run
    if [ -f "models/qnn/dinov2_vits14_fp16.dlc" ]; then
        for backend in $backends; do
            log "Benchmarking DINOv2 on $backend..."

            local be_lib="libQnn${backend}.so"
            local be_path
            be_path=$(find "$QAIRT_ROOT" -name "$be_lib" -type f 2>/dev/null | head -1)

            if [ -z "$be_path" ]; then
                warn "  ⚠️  $backend library not found"
                continue
            fi

            mkdir -p "profiling/qnn_${backend,,}"

            # Create a test input
            python3 -c "
import numpy as np
np.random.randn(1, 3, 224, 224).astype(np.float32).tofile('profiling/qnn_test_input.raw')
" 2>/dev/null
            echo "profiling/qnn_test_input.raw 0" > /tmp/qnn_test_inputs.txt

            # Run via qnn-net-run
            if command -v qnn-net-run &>/dev/null; then
                qnn-net-run \
                    --model "models/qnn/dinov2_vits14_fp16.dlc" \
                    --input_list /tmp/qnn_test_inputs.txt \
                    --backend "$be_path" \
                    --output_dir "profiling/qnn_${backend,,}/" \
                    2>&1 | tail -5
                log "  ✅ $backend inference completed"
            else
                warn "  ⚠️  qnn-net-run not in PATH — skipping automated benchmark"
            fi
        done
    fi

    # Summary
    echo ""
    log "QNN Acceleration Summary:"
    echo "  CPU baseline (from earlier benchmarks):"
    echo "    - DINOv2 (ONNX CPU):    258ms (center-crop), 708ms (full pipeline)"
    echo "    - YOLO (Ultralytics CPU): 766-2609ms"
    echo "  Expected QNN improvement:"
    echo "    - HTP (CDSP) FP16:       ~50-100ms  (5-10× speedup)"
    echo "    - HTP (CDSP) INT8:       ~30-60ms   (10-15× speedup)"
    echo "    - GPU (Adreno) FP16:     ~40-80ms   (8-12× speedup)"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║   Qualcomm QAIRT (QNN) SDK — BCS Pipeline Acceleration     ║"
    echo "║   Direct download — NO Qualcomm account needed              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    local mode="${1:-interactive}"
    local sdk_path=""
    local auto_download=false
    local convert_only=false

    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --sdk-path) sdk_path="$2"; shift 2 ;;
            --auto-download) auto_download=true; shift ;;
            --convert-only) convert_only=true; shift ;;
            *) shift ;;
        esac
    done

    check_prereqs

    if [ "$convert_only" = true ]; then
        source "$PROJECT_DIR/qnn_env.sh" 2>/dev/null || {
            err "QNN SDK not installed. Run with --auto-download first."
            exit 1
        }
        convert_models
        run_benchmark
        return
    fi

    if [ "$auto_download" = true ]; then
        sdk_path=$(download_qnn_sdk "$PROJECT_DIR/qnn_sdk")
    fi

    if [ -n "$sdk_path" ] && [ -f "$sdk_path" ]; then
        install_qnn_sdk "$sdk_path"
        install_sdk_deps
        convert_models
        run_benchmark
    else
        warn "No SDK zip provided or found."
        echo ""
        echo "  Direct download URL (no login required):"
        echo "    $QNN_DOWNLOAD_URL"
        echo ""
        echo "  Quick start:"
        echo "    bash $0 --auto-download"
        echo ""
        echo "  Or download manually and:"
        echo "    bash $0 --sdk-path /path/to/v2.48.0.260626.zip"
        echo ""
        echo "  Current CPU-only pipeline is fully functional:"
        echo "    - Center-crop:  3.3 FPS"
        echo "    - Skip 2 + half: 1.7 FPS"
        echo "  Expected with QNN CDSP: 5-15 FPS"
    fi
}

main "$@"
