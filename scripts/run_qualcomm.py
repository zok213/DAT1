import subprocess
import sys
import os
import time

def main():
    print("[Python Wrapper] Starting Qualcomm RB3 Gen2 Edge Pipeline...")
    
    executable_path = "optimization_suite/cpp/build/benchmark_runner"
    
    if not os.path.exists(executable_path):
        print(f"Error: {executable_path} not found. Did you run ./build_qualcomm.sh?")
        sys.exit(1)
        
    try:
        # Pass the native Qualcomm executable
        process = subprocess.Popen([executable_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        if process.returncode == 0:
            print("[Python Wrapper] C++ Qualcomm execution completed successfully.")
        else:
            print(f"[Python Wrapper] C++ execution failed with return code {process.returncode}")
            
    except Exception as e:
        print(f"Error executing C++ pipeline: {e}")

if __name__ == "__main__":
    main()
