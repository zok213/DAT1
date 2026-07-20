import subprocess
import sys
import os

def main():
    print("[Python Wrapper] Starting Rockchip RK3588 Cow BCS Edge Pipeline...")
    
    executable_path = "build/rk3588_cow_bcs"
    yolo_model = "models/yolov8n-seg.rknn"
    dino_model = "models/dinov2_vits14.rknn"
    
    if not os.path.exists(executable_path):
        print(f"Error: {executable_path} not found. Did you run ./build_rk3588.sh?")
        sys.exit(1)
        
    try:
        # Pass the models to the native C++ executable
        process = subprocess.Popen([executable_path, yolo_model, dino_model], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        if process.returncode == 0:
            print("[Python Wrapper] C++ execution completed successfully.")
        else:
            print(f"[Python Wrapper] C++ execution failed with return code {process.returncode}")
            
    except Exception as e:
        print(f"Error executing C++ pipeline: {e}")

if __name__ == "__main__":
    main()
