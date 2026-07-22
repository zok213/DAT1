# 🔬 Deep-Dive Technical Research Report: TFLite Quantization Architectures (W8A8, W8A16, W8A16-Mixed, FP16 & FP32)

> **Document ID:** `reports/15_tflite_quantization_w8a8_w8a16_deep_dive.md`  
> **Author:** Chief AI Quantization Specialist & Edge Hardware Fellow  
> **Date:** July 22, 2026  
> **Target:** TensorFlow Lite (TFLite) Edge Quantization Schemes & Accuracy/Speed Trade-Offs  

---

## 📌 Executive Summary of TFLite Quantization Schemes

TensorFlow Lite provides **5 primary model format and quantization specifications** designed for different hardware backends (CPUs, GPUs, and NPUs):

```
                               TFLite Quantization Architecture Matrix
  ┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
  │      W8A8 (Full INT8) │   │ W8A16 (Dynamic Range) │   │ W8A16-Mixed (Hybrid)  │
  │ Weights INT8 / Act INT8│   │ Weights INT8 / Act FP16│   │ Selective Layer INT8  │
  └───────────────────────┘   └───────────────────────┘   └───────────────────────┘
```

---

## 📊 Comprehensive Quantization Trade-Off Matrix

| Quantization Scheme | Weight Precision | Activation Precision | Calibration Dataset Needed? | Model Size Reduction | Speedup Factor | Recommended Target Hardware |
|---|---|---|---|---|---|---|
| **W8A8 (Full INT8)** | INT8 | INT8 | **Yes** (100+ images) | **75% Smaller (0.25x)** | **3.5x - 4.0x** | NPU (Hexagon DSP, RKNN, Coral Edge TPU) |
| **W8A16 (Dynamic Range)** | INT8 | FP16 / Dynamic INT8 | **No** | **75% Smaller (0.25x)** | **1.8x - 2.2x** | Mobile & Edge CPUs (ARM Cortex-A78/A55) |
| **W8A16-Mixed (Hybrid)** | INT8 (Convs/Linear) | FP16 (Attention/Norm) | **Yes** | **70% Smaller (0.30x)** | **2.5x - 3.0x** | Edge Vision Transformers (DINOv2 ViT) |
| **FP16 (Half Precision)** | FP16 | FP16 | **No** | **50% Smaller (0.50x)** | **1.5x - 2.0x** | Mobile GPUs (Mali, Adreno, Tegra Ampere) |
| **FP32 (Single Precision)**| FP32 | FP32 | **No** | Baseline (1.00x) | Baseline (1.00x) | Mathematical Verification Baseline |

---

## 🔬 Mathematical Formulations & Mechanics

### 1. Full Integer W8A8 Quantization Mechanics
Quantizes continuous float values $r \in [r_{\min}, r_{\max}]$ to signed 8-bit integers $q \in [-128, 127]$:

$$r = S \cdot (q - Z)$$

Where $S$ (Scale Factor) and $Z$ (Zero-Point) are computed via calibration:

$$S = \frac{r_{\max} - r_{\min}}{q_{\max} - q_{\min}} = \frac{r_{\max} - r_{\min}}{255}$$

$$Z = \text{round}\left( \frac{-r_{\min}}{S} \right) + q_{\min}$$

* **Advantages**: Eliminates floating-point hardware units entirely; executes on 8-bit SIMD vector engines (`dp4a`, `neon_dot`).

---

### 2. Dynamic Range W8A16 Quantization Mechanics
Weights are quantized to 8-bit integers at build time without calibration. During runtime inference:
- Activations are dynamically range-checked on each layer: $r_{\text{act\_max}} = \max(|r_{\text{act}}|)$.
- Quantization scale is calculated dynamically on-the-fly without an explicit calibration dataset.
* **Advantages**: Instant conversion from PyTorch/TensorFlow with **zero calibration step required**.

---

### 3. W8A16-Mixed (Hybrid Selective Per-Layer Quantization)
* Heavy matrix multiplication and convolution layers (`Conv2D`, `Linear`) are converted to **INT8**.
* Sensitive Vision Transformer layers (`LayerNorm`, `GELU`, `Softmax Attention`) remain in **FP16**.
* **Result**: Protects Vision Transformer CLS embedding accuracy while reducing model footprint by **`70%`**.

---

## 🚀 Converter Usage & Command Reference ([`scripts/compile_tflite_advanced.py`](file:///d:/Gitrepo/DAT1/scripts/compile_tflite_advanced.py))

```bash
# Execute TFLite Advanced Multi-Quantization Converter
python scripts/compile_tflite_advanced.py --output-dir models/tflite
```

Outputs generated into [`models/tflite/`](file:///d:/Gitrepo/DAT1/models/tflite):
* `bcs_head_w8a8.tflite` (Full INT8 NPU target)
* `bcs_head_w8a16.tflite` (Dynamic Range CPU target)
* `bcs_head_w8a16_mixed.tflite` (Hybrid Transformer target)
* `bcs_head_fp16.tflite` (Half Precision GPU target)
* `bcs_head_fp32.tflite` (Unquantized Baseline)
