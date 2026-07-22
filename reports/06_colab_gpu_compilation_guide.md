# ☁️ Free Google Colab GPU Compilation Guide

> **Document ID:** `reports/06_colab_gpu_compilation_guide.md`  
> **Author:** Senior Edge AI Engineer  
> **Target:** Free Cloud Compilation via Google Colab (Tesla T4 / A100 GPU)  

---

## 💡 Can We Use Google Colab's Free GPU for Model Compilation?

### **YES! Absolutely.**
Building hardware-optimized model engines (such as **NVIDIA TensorRT `.engine`**, **TFLite INT8**, or **RKNN `.rknn`** models) requires heavy CUDA compute, high system RAM, and platform toolkits that are tedious to install locally.

Google Colab offers **free access to NVIDIA Tesla T4 GPUs** with CUDA pre-configured, making it the ideal cloud environment to compile edge model binaries for free.

---

## 🚀 1-Click Compilation Workflow on Google Colab

### **Step 1: Open Google Colab Notebook**
Open [`notebooks/colab_gpu_model_compiler.ipynb`](file:///d:/Gitrepo/DAT1/notebooks/colab_gpu_model_compiler.ipynb) in Google Colab or run the commands below in any Colab GPU runtime:

```python
# 1. Enable GPU Runtime in Google Colab (Runtime -> Change runtime type -> T4 GPU)
!nvidia-smi

# 2. Install Edge Compiler Toolkits
!pip install ultralytics torch torchvision onnx onnxruntime-gpu tensorflow rknn-toolkit2

# 3. Clone repository
!git clone https://github.com/zok213/DAT1.git
%cd DAT1

# 4. Compile All Models (YOLOv8, DINOv2, BcsHead)
!python scripts/compile_all_models.py --output-dir compiled_models/ --quantize int8

# 5. Compile TFLite FP32, FP16, and INT8 Variants
!python scripts/compile_to_tflite.py --input compiled_models/bcs_head.onnx --output-dir compiled_models/tflite/ --quantize all
!python scripts/compile_to_tflite.py --input compiled_models/dinov2_vits14.onnx --output-dir compiled_models/tflite/ --quantize fp16

# 6. Download All Compiled Binaries (.zip)
from google.colab import files
!zip -r compiled_edge_models.zip compiled_models/
files.download('compiled_edge_models.zip')
```

---

## 📊 Compiled Model Formats Generated

| Target Architecture | Compiled File Format | Precision | Compiler Tool / Library |
|---|---|---|---|
| **NVIDIA Jetson Orin** | `yolov8n_seg.engine` | INT8 / FP16 | TensorRT 10.x |
| **Radxa CM5 (RK3588)** | `dinov2_vits14.rknn` | INT8 / FP16 | RKNN-Toolkit2 |
| **Qualcomm RB3 Gen2** | `dinov2_fp16_net.json` | FP16 / INT8 | QNN / QAIRT SDK |
| **General Edge / Mobile**| `bcs_head_int8.tflite` | INT8 (Calibrated)| TensorFlow Lite |
| **General Edge / Mobile**| `dinov2_vits14_fp16.tflite` | FP16 | TensorFlow Lite |

---

## 💡 Best Practice Tip for Edge Deployment
After downloading `compiled_edge_models.zip` from Google Colab, extract the binaries into the `models/` directory of your hardware targets ([`jetson_orin_nano/`](file:///d:/Gitrepo/DAT1/jetson_orin_nano), [`radxa_cm5/`](file:///d:/Gitrepo/DAT1/radxa_cm5), or [`qualcomm_adaptation/`](file:///d:/Gitrepo/DAT1/qualcomm_adaptation)) to immediately run hardware-accelerated zero-copy inference!
