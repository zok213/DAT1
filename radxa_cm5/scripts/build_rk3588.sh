#!/usr/bin/env bash
set -e

echo "================================================="
echo " Building Radxa CM5 (RK3588) C++ BCS Engine      "
echo "================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

BUILD_DIR="$ROOT_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo "[SUCCESS] Build finished! Binary located at $BUILD_DIR/rk3588_cow_bcs"
