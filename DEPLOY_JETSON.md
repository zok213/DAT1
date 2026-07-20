# Deploying BCS to Jetson Orin NX — step by step

Pipeline: **mp4 → YOLOv8-seg (detect+mask cow) → DINOv2 ViT-S (feature) → BcsHead (softmax) → overlay**.
Backbone is ViT-S because our benchmark showed ViT-S ≈ ViT-B (bootstrap CI includes 0) at ~4× lower cost —
the right edge choice, backed by the science.

## 0. Files to copy to the Jetson
From the host (`A:/dat301/bcs_project`), `scp` these into one folder on the Orin (e.g. `~/bcs`):
```
deploy/jetson_bcs_demo.py
deploy/production_head_vits.pt
deploy/production_config.json
deploy/dinov2_vits14.onnx          # for the TRT engine
pipeline/yolov8n-seg.pt            # YOLO weights
<a test>.mp4                       # a cow video to run on
```
Example (run in Windows Git-bash, fill in your Jetson user/ip):
```bash
scp deploy/jetson_bcs_demo.py deploy/production_head_vits.pt deploy/production_config.json \
    deploy/dinov2_vits14.onnx pipeline/yolov8n-seg.pt  user@JETSON_IP:~/bcs/
```

## 1. Max performance (once per boot)
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
sudo nvpmodel -q | grep -i power     # expect MAXN
```

## 2. Dependencies  — YOUR SYSTEM: JetPack 6.1 (L4T r36.4), CUDA 12.6, TensorRT 10.3, Python 3.10, 8 GB
TensorRT + OpenCV present; ultralytics 8.4.26 present but dead without torch. The Orin could **not reach
`jetson-ai-lab.dev`**, so we install torch **offline from files** (already downloaded to your PC in
`deploy/wheels/`). Never `pip install torch` from plain PyPI — it's x86/CPU and breaks CUDA.

**2a. Copy the wheels (from your Windows PC) to the Orin:**
```bash
scp deploy/wheels/*.whl deploy/wheels/*.tar.gz   aitlab@JETSON_IP:~/bcs/
```

**2b. Install numpy<2 + torch from the wheels (fully offline — avoids the Orin's IPv6/PyPI quirk):**
```bash
cd ~/bcs
pip3 install ./numpy-1.26.4-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install ./torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"  # must say cuda True
```
👉 If `cuda True`, you can **immediately** validate the core pipeline (step 3) — no torchvision yet.

**2c. Build torchvision 0.20.0 against this torch (needed only for YOLO; ~20-40 min on the Orin):**
```bash
sudo apt-get install -y libjpeg-dev zlib1g-dev ninja-build      # image ops + faster build
tar xf torchvision-0.20.0-src.tar.gz && cd vision-0.20.0
export BUILD_VERSION=0.20.0 FORCE_CUDA=1
python3 setup.py install --user
cd ~/bcs
python3 -c "import torchvision;print('tv',torchvision.__version__)"
```
> `ultralytics` (8.4.26) is already installed and will now work once torchvision imports.
> No pycuda needed — the DINOv2 TRT wrapper uses torch CUDA tensors as I/O buffers.
> 8 GB module: keep TRT batch=1 (already the case) and close other apps while building engines.

> **If the Orin *does* have internet** (check: `getent hosts github.com` resolves), you can skip the scp
> and instead `git clone --branch v0.20.0 https://github.com/pytorch/vision` for 2c. torch still installs
> from the wheel file either way.

## 3. Build the DINOv2 TensorRT FP16 engine (needs only trtexec — do right after 2b)
TRT engines are hardware-specific → build **on the Orin**. TensorRT 10 uses `--memPoolSize` (not `--workspace`):
```bash
cd ~/bcs
/usr/src/tensorrt/bin/trtexec --onnx=dinov2_vits14.onnx --fp16 \
    --saveEngine=dinov2_vits14_fp16.engine --memPoolSize=workspace:2048
# perf read (look at mean latency / Throughput):
/usr/src/tensorrt/bin/trtexec --loadEngine=dinov2_vits14_fp16.engine
```
> If it gets OOM-killed on the 8 GB module, drop to `--memPoolSize=workspace:1024`.
> ✅ `jetson_bcs_demo.py:DinoTRT` is already written for **TensorRT 10.3** (`execute_async_v3` +
> `set_tensor_address`, torch tensors as I/O) — matches your system, no changes needed.

## 4. Bring-up test — DINOv2(TRT)+head only, NO YOLO (torch-only; run before torchvision finishes)
This proves the core scoring path with just torch + the TRT engine (`--no-detect` scores a center crop):
```bash
python3 jetson_bcs_demo.py --video test.mp4 --out test_nodetect.mp4 --no-detect \
    --dino-engine dinov2_vits14_fp16.engine \
    --head production_head_vits.pt --config production_config.json
```
End line `end2end X FPS | ... DINO .. ms` = your DINOv2+head speed. Send me that number.

## 5. Full pipeline — add YOLO (after torchvision from 2c is built)
Export the YOLO engine (ultralytics-native, needs torch+torchvision), then run the real pipeline:
```bash
yolo export model=yolov8n-seg.pt format=engine half=True       # -> yolov8n-seg.engine
python3 jetson_bcs_demo.py --video test.mp4 --out test_bcs.mp4 \
    --yolo yolov8n-seg.engine --dino-engine dinov2_vits14_fp16.engine \
    --head production_head_vits.pt --config production_config.json --display
```
(Drop `--dino-engine` to fall back to PyTorch-FP16 DINOv2 — needs torch.hub internet once; the TRT path is preferred.)

## 6. Tuning knobs (in priority order)
1. **Batch the cows** through DINOv2 (already batched per frame) — biggest win when multiple cows/frame.
2. **INT8** engine (`trtexec --int8` + a calibration set of ~200 crops) — ~2× over FP16, needs an accuracy re-check.
3. **Frame skipping / ROI** — score every Nth frame or only when a cow is newly detected (BCS is slow-changing).
4. **Hardware video decode** — read mp4 via GStreamer (`nvv4l2decode`) instead of OpenCV CPU decode for high-res.

## 7. ⚠️ The honesty note (do NOT skip this in the demo)
The overlay prints **"OOD: screening only"** on purpose. Our deployment-gap analysis found real-CCTV is a
**separate domain** (linear-probe separability 1.0 vs training, model over-confident off-domain). The head's
honest QWK is `CowDatabase 0.639 / CowDB 0.486 / CowDatabase2 −0.046 / pooled 0.34` — **screening-grade**,
and only *within* the research datasets. On genuinely new CCTV footage, treat outputs as a **coarse screen**,
always show the confidence, and validate against a few human-scored frames before trusting it. Closing this
gap needs real CCTV labels + a trained domain-adaptation model (see `results/da_baseline.json` for the
unsupervised baseline that shrinks the *feature* gap but not, on its own, the accuracy).

## 8. Known failure point to watch
`yolov8n-seg` is COCO-trained; **top-down cows are hard for it** (COCO cows are side-on). If detection misses,
lower `--conf`, or fine-tune YOLO on a few annotated barn frames. For side/oblique research-style video it's fine.
```
