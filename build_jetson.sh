#!/bin/bash
set -e

echo "=========================================="
echo " Building NVIDIA Jetson Orin DeepStream   "
echo "=========================================="

mkdir -p build
cd build
cmake ..
make -j4

echo "=========================================="
echo " Build successful!                        "
echo " Execute: ./build/jetson_cow_bcs          "
echo "=========================================="
