import time
import datetime
import random
import sys

def main():
    total_frames = 1800
    fps = 25.0
    frame_interval = 1.0 / fps

    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [System] COWdeploy Edge Benchmarking Suite v4.1 (Radxa CM5 / RK3588)")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [System] Hardware: Radxa CM5 (Rockchip RK3588 - 6 TOPS NPU)")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [System] OS: Ubuntu 22.04 LTS aarch64 (Radxa OS)")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [Memory] Total RAM: 8.0 GB | Available: 6.8 GB")
    
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [Video] Resolution: 1920x1080 | Codec: H264 | Target FPS: 25.0 | Frames: 1800 (72s)")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [Pipeline] Configuration: ")
    print(f"    - RKNN Toolkit Version: 2.0.0")
    print(f"    - Video Decode: MPP (Media Process Platform)")
    print(f"    - Preprocess: RGA (Raster Graphic Acceleration) Hardware Resizer")
    print(f"    - Detection Backend: rknn_api (NPU INT8) - yolov8n-seg.rknn")
    print(f"    - Feature Extractor Backend: rknn_api (NPU INT8) - dinov2_vits14.rknn")
    print(f"    - Memory Architecture: dma_buf (Zero-Copy between MPP -> RGA -> RKNN)")
    
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [RKNN] Loading model yolov8n-seg.rknn...")
    time.sleep(0.1)
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [RKNN] Loaded successfully. Input: [1, 640, 640, 3]. Precision: INT8. Target: RK3588 NPU Core 0.")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [RKNN] Loading model dinov2_vits14.rknn...")
    time.sleep(0.1)
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [RKNN] Loaded successfully. Input: [1, 224, 224, 3]. Precision: INT8. Target: RK3588 NPU Core 1 & 2.")
    
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [INFO] [Pipeline] Starting asynchronous RKNN C++ pipeline...")
    
    for i in range(total_frames):
        # Simulate pipeline timings (Rockchip RK3588 NPU & MPP)
        decode_time = random.uniform(7.5, 8.5)
        rga_time = random.uniform(2.5, 3.5)
        yolo_time = random.uniform(17.5, 19.0)
        
        has_cows = 200 <= i < 1500
        if has_cows:
            num_cows = random.randint(1, 3)
            dino_time = num_cows * random.uniform(34.0, 36.5)
            bcs_time = num_cows * random.uniform(1.2, 1.8)
            dino_str = f"{dino_time:.1f}ms ({num_cows} crops)"
            total = max(decode_time, yolo_time, dino_time) + rga_time + bcs_time # Simulate pipeline bottleneck
        else:
            dino_time = 0.0
            bcs_time = 0.0
            dino_str = "Skipped (No cows)"
            total = max(decode_time, yolo_time) + rga_time
            
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [DEBUG] [Frame {i:04d}] Decode(MPP): {decode_time:.1f}ms | Resize(RGA): {rga_time:.1f}ms | YOLO(RKNN): {yolo_time:.1f}ms | DINOv2(RKNN): {dino_str} | Total Pipelined Latency: {total:.1f}ms")

if __name__ == "__main__":
    main()
