#!/bin/bash
set -e

echo "=========================================="
echo " Building Rockchip RK3588 (Radxa CM5) App "
echo "=========================================="

mkdir -p build
cd build
cmake ..
make -j4

echo "=========================================="
echo " Build successful!                        "
echo " Execute: ./build/rk3588_cow_bcs          "
echo "=========================================="
