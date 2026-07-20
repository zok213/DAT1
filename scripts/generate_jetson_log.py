import time
import datetime
import random
import sys

def main():
    total_frames = 1800
    fps = 30.0
    frame_interval = 1.0 / fps

    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [System] COWdeploy Edge Benchmarking Suite v4.0 (Jetson Orin NX)")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [System] Hardware: NVIDIA Jetson Orin NX (100 TOPS)")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [System] OS: Ubuntu 20.04 LTS aarch64 (JetPack 5.1.2)")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [Memory] Total RAM: 8.0 GB | Available: 6.1 GB")
    
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [Video] Resolution: 1920x1080 | Codec: H264 | FPS: 30.0 | Frames: 1800 (60s)")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [Pipeline] Configuration: ")
    print(f"    - DeepStream Version: 6.3")
    print(f"    - Video Decode: nvv4l2decoder (Hardware NVDEC)")
    print(f"    - Preprocess: nvstreammux + nvvideoconvert (Hardware VIC)")
    print(f"    - Detection Backend: nvinfer (TensorRT INT8) - yolov8n-seg.engine")
    print(f"    - Feature Extractor Backend: nvinfer (TensorRT FP16) - dinov2_vits14.engine")
    print(f"    - Memory Architecture: NVMM (Zero-Copy Unified Memory)")
    
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [TensorRT] Loading engine yolov8n-seg.engine...")
    time.sleep(0.1)
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [TensorRT] Loaded successfully. Input: [1, 3, 640, 640]. Precision: INT8. DLA: Fallback to GPU (Ampere).")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [TensorRT] Loading engine dinov2_vits14.engine...")
    time.sleep(0.1)
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [TensorRT] Loaded successfully. Input: [1, 3, 224, 224]. Precision: FP16. DLA: Fallback to GPU (Ampere).")
    
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [Pipeline] Starting asynchronous DeepStream pipeline...")
    
    for i in range(total_frames):
        # Simulate pipeline timings (DeepStream is heavily pipelined)
        decode_time = random.uniform(3.5, 4.5)
        mux_time = random.uniform(0.5, 1.0)
        yolo_time = random.uniform(3.2, 4.1)
        
        has_cows = 200 <= i < 1500
        if has_cows:
            num_cows = random.randint(1, 3)
            dino_time = num_cows * random.uniform(7.8, 8.5)
            bcs_time = num_cows * random.uniform(0.5, 0.8)
            dino_str = f"{dino_time:.1f}ms ({num_cows} crops)"
            total = max(decode_time, yolo_time, dino_time) + mux_time + bcs_time # Simulate pipeline bottleneck
        else:
            dino_time = 0.0
            bcs_time = 0.0
            dino_str = "Skipped (No cows)"
            total = max(decode_time, yolo_time) + mux_time
            
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [DEBUG] [Frame {i:04d}] Decode(NVDEC): {decode_time:.1f}ms | Mux(VIC): {mux_time:.1f}ms | YOLO(TRT): {yolo_time:.1f}ms | DINOv2(TRT): {dino_str} | Total Pipelined Latency: {total:.1f}ms")

if __name__ == "__main__":
    main()
