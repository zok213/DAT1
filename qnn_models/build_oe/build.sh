#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
QNN_ROOT="${QNN_ROOT:-/home/ubuntu/COWdeploy/qnn_sdk/qairt/2.48.0.260626}"
MODEL_NAME="dinov2_fp32"
MODEL_CPP="${PROJECT_DIR}/build_fp32/${MODEL_NAME}.cpp"
MODEL_BIN="${PROJECT_DIR}/${MODEL_NAME}.bin"

# Cross-compiler selection.
# Ubuntu:  CROSS_COMPILE=aarch64-linux-gnu-  (default)
# OE:      CROSS_COMPILE=aarch64-oe-linux-   + SYSROOT=/path/to/sysroots/armv8a-oe-linux
CROSS_COMPILE="${CROSS_COMPILE:-aarch64-linux-gnu-}"
SYSROOT="${SYSROOT:-}"

CXX="${CROSS_COMPILE}g++"
OBJCOPY="${CROSS_COMPILE}objcopy"

CXXFLAGS="-std=c++17 -fno-rtti -fPIC -Wall -O2 -Wno-write-strings -fvisibility=hidden"
LDFLAGS="-shared -s -fPIC -fvisibility=hidden"

if [ -n "$SYSROOT" ]; then
    CXXFLAGS="$CXXFLAGS --sysroot=$SYSROOT"
    LDFLAGS="$LDFLAGS --sysroot=$SYSROOT"
fi

INCLUDES="-I${QNN_ROOT}/include/QNN"
INCLUDES+=" -I${QNN_ROOT}/share/QNN/converter/jni"
INCLUDES+=" -I${QNN_ROOT}/share/QNN/converter/jni/linux"

BINARY_RAW_DIR="${SCRIPT_DIR}/obj/binary"
BINARY_O_DIR="${SCRIPT_DIR}/obj/binary/oe"
OBJ_DIR="${SCRIPT_DIR}/obj"
LIBS_DIR="${SCRIPT_DIR}/libs"

CONVERTER_DIR="${QNN_ROOT}/share/QNN/converter/jni"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     QNN Model Library Build                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Target triple:  ${CROSS_COMPILE%%-}"
echo "  Compiler:       $(which $CXX 2>/dev/null || echo 'NOT FOUND')"
echo "  SYSROOT:        ${SYSROOT:-(none)}"
echo "  Model .cpp:     $MODEL_CPP"
echo "  Model .bin:     $MODEL_BIN"
echo ""

if ! command -v "$CXX" &>/dev/null; then
    echo "ERROR: Cross-compiler '$CXX' not found."
    echo ""
    echo "For aarch64-oe-linux (Yocto/OE on RB3gen2):"
    echo "  Install the Qualcomm Yocto SDK, then:"
    echo "    export CROSS_COMPILE=aarch64-oe-linux-"
    echo "    export SYSROOT=/path/to/sysroots/armv8a-oe-linux"
    echo "    export PATH=\$PATH:/path/to/sysroots/x86_64-qtisdk-linux/usr/bin/aarch64-oe-linux/"
    echo ""
    echo "Available from: https://www.qualcomm.com/developer/software/qualcomm-linux-sdk"
    exit 1
fi

create_binary_objects() {
    echo "  [1/4] Creating binary weight object files..."
    mkdir -p "$BINARY_O_DIR"
    for rawfile in "$BINARY_RAW_DIR"/*.raw; do
        [ -f "$rawfile" ] || continue
        base=$(basename "$rawfile" .raw)
        $OBJCOPY -B aarch64 -I binary -O elf64-littleaarch64 "$rawfile" "${BINARY_O_DIR}/${base}.o"
    done
    echo "        -> $(ls "$BINARY_O_DIR"/*.o 2>/dev/null | wc -l) .o files"
}

compile_helpers() {
    echo "  [2/4] Compiling QNN helper sources..."
    mkdir -p "$OBJ_DIR"
    $CXX $CXXFLAGS $INCLUDES -c "$CONVERTER_DIR/QnnModel.cpp"      -o "${OBJ_DIR}/QnnModel.o"      -w
    $CXX $CXXFLAGS $INCLUDES -c "$CONVERTER_DIR/linux/QnnModelPal.cpp" -o "${OBJ_DIR}/QnnModelPal.o" -w
    $CXX $CXXFLAGS $INCLUDES -c "$CONVERTER_DIR/QnnWrapperUtils.cpp"   -o "${OBJ_DIR}/QnnWrapperUtils.o" -w
}

compile_model() {
    echo "  [3/4] Compiling ${MODEL_NAME}.cpp..."
    $CXX $CXXFLAGS $INCLUDES -c "$MODEL_CPP" -o "${OBJ_DIR}/${MODEL_NAME}.o"
}

link_library() {
    echo "  [4/4] Linking shared library..."
    mkdir -p "$LIBS_DIR"
    $CXX $LDFLAGS \
        -o "${LIBS_DIR}/lib${MODEL_NAME}.so" \
        "${OBJ_DIR}/QnnModel.o" \
        "${OBJ_DIR}/QnnModelPal.o" \
        "${OBJ_DIR}/QnnWrapperUtils.o" \
        "${OBJ_DIR}/${MODEL_NAME}.o" \
        "${BINARY_O_DIR}"/*.o
}

main() {
    create_binary_objects
    compile_helpers
    compile_model
    link_library
    echo ""
    echo "  Build complete: ${LIBS_DIR}/lib${MODEL_NAME}.so"
    ls -lh "${LIBS_DIR}/lib${MODEL_NAME}.so"
    file "${LIBS_DIR}/lib${MODEL_NAME}.so"
    echo ""
    readelf -d "${LIBS_DIR}/lib${MODEL_NAME}.so" 2>/dev/null | grep NEEDED || true
}

main "$@"
