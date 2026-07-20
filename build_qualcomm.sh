#!/bin/bash
set -e

echo "=========================================="
echo " Building Qualcomm RB3 Gen2 C++ Pipeline  "
echo "=========================================="

cd optimization_suite/cpp
mkdir -p build
cd build
cmake ..
make -j4

echo "=========================================="
echo " Build successful!                        "
echo " Execute: ./optimization_suite/cpp/build/benchmark_runner "
echo "=========================================="
