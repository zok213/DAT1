#!/bin/bash
source venv/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python3 profiling/profiler.py \
    --video sample_cow_video.mp4 \
    --yolo models/yolo_tflite/yolov8n-seg_float32.tflite \
    --dino-onnx dinov2_vits14.onnx \
    --head production_head_vits.pt \
    --config production_config.json \
    --max-frames 100 --warmup 150 \
    --json
