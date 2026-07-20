import subprocess
import sys
import os

def main():
    print("[Python Wrapper] Starting Jetson Orin DeepStream Edge Pipeline...")
    
    executable_path = "build/jetson_cow_bcs"
    
    if not os.path.exists(executable_path):
        print(f"Error: {executable_path} not found. Did you run ./build_jetson.sh?")
        sys.exit(1)
        
    try:
        # Pass the native DeepStream executable
        process = subprocess.Popen([executable_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        if process.returncode == 0:
            print("[Python Wrapper] C++ DeepStream execution completed successfully.")
        else:
            print(f"[Python Wrapper] C++ execution failed with return code {process.returncode}")
            
    except Exception as e:
        print(f"Error executing C++ pipeline: {e}")

if __name__ == "__main__":
    main()
