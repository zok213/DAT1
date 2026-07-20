"""
Jetson Orin NX — live-annotated BCS demo on pre-recorded video.

Pipeline per frame:
  frame -> YOLOv8-seg (detect+mask cows) -> crop each cow -> resize 224 -> DINOv2 ViT-S (CLS 384)
        -> BcsHead (softmax) -> band + confidence -> overlay box + label -> write annotated mp4

DINOv2 backend:
  --dino-engine PATH  : TensorRT FP16 engine (fastest; needs tensorrt+pycuda, version-specific)
  (default)           : PyTorch FP16 via torch.hub (version-proof; still GPU-accelerated)

YOLO backend: ultralytics loads either yolov8n-seg.pt OR a prebuilt .engine transparently.

Example:
  python3 jetson_bcs_demo.py --video barn.mp4 --out barn_bcs.mp4 \
      --yolo yolov8n-seg.engine --head production_head_vits.pt --config production_config.json --display

Deps on Jetson: ultralytics, torch (NVIDIA Jetson wheel), opencv, numpy. (+ tensorrt,pycuda for --dino-engine)
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import cv2
import torch
import torch.nn as nn

COCO_COW = 19                                   # 'cow' class id in COCO
BAND_COLORS = {0: (60, 76, 231), 1: (104, 168, 85), 2: (82, 78, 196)}  # BGR: thin/ideal/fat


# ── the production head (must match produce_production_head.py) ──────────────
class BcsHead(nn.Module):
    def __init__(self, in_dim=384, d=128, p=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(p))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(p))
        self.cls = nn.Linear(d, 3)

    def forward(self, x):
        return self.cls(self.head(self.proj(x)))


# ── DINOv2 backends ─────────────────────────────────────────────────────────
class DinoTorch:
    """PyTorch FP16 DINOv2 ViT-S/14 (version-proof default)."""
    def __init__(self, device):
        self.device = device
        self.m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval().half()

    @torch.no_grad()
    def __call__(self, batch_chw: torch.Tensor) -> np.ndarray:  # (B,3,224,224) fp32 -> (B,384)
        return self.m(batch_chw.to(self.device).half()).float().cpu().numpy()


class DinoTRT:
    """TensorRT 10 FP16 engine (JetPack 6 / TRT 10.3). Uses torch CUDA tensors as I/O buffers
    (set_tensor_address + execute_async_v3) so no pycuda is needed — torch is already installed."""
    def __init__(self, engine_path, device="cuda"):
        import tensorrt as trt
        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        # discover the single input / single output tensor names
        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            nm = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
                self.in_name = nm
            else:
                self.out_name = nm
        self.d_in = torch.empty((1, 3, 224, 224), dtype=torch.float32, device=device)
        self.d_out = torch.empty((1, 384), dtype=torch.float32, device=device)
        self.ctx.set_tensor_address(self.in_name, self.d_in.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.d_out.data_ptr())

    def __call__(self, batch_chw: torch.Tensor) -> np.ndarray:
        outs = []
        stream = torch.cuda.current_stream().cuda_stream
        for i in range(batch_chw.shape[0]):        # engine is batch=1; loop over cows
            self.d_in.copy_(batch_chw[i:i + 1].to(self.device))
            self.ctx.execute_async_v3(stream_handle=stream)
            torch.cuda.synchronize()
            outs.append(self.d_out.clone())
        return torch.cat(outs).cpu().numpy()


# ── preprocessing ───────────────────────────────────────────────────────────
def make_crop(frame, box, mask, mean, std, size=224):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if mask is not None:                            # zero out background using the seg mask
        m = mask[y1:y2, x1:x2]
        crop = crop * m[..., None]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(mean)) / np.array(std)
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()   # (3,224,224)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="bcs_annotated.mp4")
    ap.add_argument("--yolo", default="yolov8n-seg.pt", help=".pt or prebuilt .engine")
    ap.add_argument("--dino-engine", default=None, help="TRT FP16 engine; omit = PyTorch FP16")
    ap.add_argument("--head", default="production_head_vits.pt")
    ap.add_argument("--config", default="production_config.json")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--no-detect", action="store_true",
                    help="skip YOLO; score a center-square crop of the whole frame "
                         "(torch + TRT engine only, no ultralytics/torchvision) — for bring-up")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(args.config))
    mean, std = cfg["preprocess"]["mean"], cfg["preprocess"]["std"]
    classes = cfg["classes"]

    if args.no_detect:
        yolo = None
        print("[init] --no-detect: scoring whole-frame center crop (no YOLO)")
    else:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo)
    dino = DinoTRT(args.dino_engine, device) if args.dino_engine else DinoTorch(device)
    head = BcsHead().to(device).eval()
    head.load_state_dict(torch.load(args.head, map_location=device))
    print(f"[init] device={device} dino={'TRT' if args.dino_engine else 'torch-fp16'} "
          f"head_qwk={cfg.get('honest_qwk_grouped5fold_perview')}")

    cap = cv2.VideoCapture(args.video)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))

    n, t_yolo, t_dino, t_all = 0, 0.0, 0.0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        t0 = time.time()
        crops, boxes = [], []
        if yolo is None:                                    # --no-detect: whole-frame center square
            s = min(H, W); cx, cy = W // 2, H // 2
            box = np.array([cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2], np.float32)
            c = make_crop(frame, box, None, mean, std)
            if c is not None:
                crops.append(c); boxes.append(box)
            t1 = time.time(); t_yolo += t1 - t0
        else:
            r = yolo.predict(frame, classes=[COCO_COW], conf=args.conf, verbose=False)[0]
            t1 = time.time(); t_yolo += t1 - t0
            if r.boxes is not None and len(r.boxes):
                masks = r.masks.data.cpu().numpy() if r.masks is not None else [None] * len(r.boxes)
                for b, mk in zip(r.boxes, masks):
                    box = b.xyxy[0].cpu().numpy()
                    m = None
                    if mk is not None:
                        m = cv2.resize(mk.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                    c = make_crop(frame, box, m, mean, std)
                    if c is not None:
                        crops.append(c); boxes.append(box)

        if crops:
            batch = torch.stack(crops)
            t2 = time.time()
            feats = dino(batch)                                  # (K,384)
            t3 = time.time(); t_dino += t3 - t2
            with torch.no_grad():
                logits = head(torch.tensor(feats, device=device))
                prob = torch.softmax(logits, 1).cpu().numpy()
            for box, p in zip(boxes, prob):
                k = int(p.argmax()); conf = float(p[k])
                x1, y1, x2, y2 = [int(v) for v in box]
                col = BAND_COLORS[k]
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                label = f"BCS: {classes[k]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + 11 * len(label), y1), col, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        t_all += time.time() - t0
        n += 1
        inst_fps = n / t_all
        cv2.putText(frame, f"{inst_fps:4.1f} FPS  (OOD: screening only)", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        if args.display:
            cv2.imshow("BCS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release(); writer.release(); cv2.destroyAllWindows()
    if n:
        print(f"[done] {n} frames -> {args.out}")
        print(f"[perf] end2end {n/t_all:5.1f} FPS | YOLO {1000*t_yolo/n:5.1f} ms | "
              f"DINO {1000*t_dino/n:5.1f} ms/frame(all cows)")


if __name__ == "__main__":
    main()
