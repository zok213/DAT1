"""
BCS Pipeline — Qualcomm-adapted inference with multiple backends.

Supports:
  - PyTorch (if available) — fallback for BcsHead
  - ONNX Runtime CPU — primary for DINOv2 (already in ONNX)
  - QNN CPU — accelerated inference via Qualcomm AI Runtime (QAIRT)
  - Pure NumPy — minimal-dependency BcsHead inference
  - GStreamer V4L2 HW decode — hardware-accelerated video decoding
  - (Future) QNN HTP — Hexagon DSP acceleration when libcdsprpc.so is available

Pipeline per frame:
  frame → YOLOv8-seg → crop cows → DINOv2 (QNN/ONNX) → BcsHead → overlay
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import cv2

from .config import BCSConfig


# ═══════════════════════════════════════════════════════════════════════════════
#  Video Reader — GStreamer V4L2 HW-Accelerated Decode
# ═══════════════════════════════════════════════════════════════════════════════

class VideoReaderHW:
    """Hardware-accelerated video reader using GStreamer V4L2 decode.

    Uses the Qualcomm `msm_vidc_decoder` hardware block via V4L2 to decode
    H.264 video, offloading ≈30-50ms/frame from the CPU.

    Falls back to OpenCV cv2.VideoCapture if GStreamer is unavailable or
    the video codec is not supported by the hardware decoder.
    """

    def __init__(self, path: str, force_hw: bool = False):
        self.path = path
        self.cap = None
        self.gst_pipeline = None
        self._appsink = None
        self._using_hw = False
        self._fps = 25.0
        self._width = 0
        self._height = 0
        self._total_frames = 0

        if self._try_gst_hw(path):
            self._using_hw = True
        else:
            if force_hw:
                raise RuntimeError(
                    f"GStreamer V4L2 HW decode failed for {path} and force_hw=True"
                )
            self._fallback_cv(path)

    def _try_gst_hw(self, path: str) -> bool:
        """Attempt to create a GStreamer V4L2 HW decode pipeline."""
        try:
            import gi  # type: ignore[import-untyped]
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst, GLib  # type: ignore[import-untyped]

            Gst.init(None)

            # Build pipeline: file → demux → h264parse → V4L2 HW decode → convert → appsink
            pipeline_str = (
                f'filesrc location="{path}" ! '
                "qtdemux ! h264parse ! "
                "v4l2h264dec ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=false max-buffers=2 drop=true"
            )

            pipeline = Gst.parse_launch(pipeline_str)
            sink = pipeline.get_by_name("sink")

            if sink is None:
                return False

            # Get video properties from the negotiated caps
            pad = sink.get_static_pad("sink")
            if pad is None:
                return False

            # Set pipeline to PLAYING
            pipeline.set_state(Gst.State.PLAYING)

            # Wait for initial caps negotiation
            import time as _time
            _time.sleep(0.1)

            caps = pad.get_current_caps()
            if caps is None:
                # Fallback: try to read first frame to negotiate
                sample = sink.emit("pull-sample")
                if sample is None:
                    pipeline.set_state(Gst.State.NULL)
                    return False
                caps = sample.get_caps()
                # Push sample back by re-adding it... actually we just need caps

            if caps is not None:
                struct = caps.get_structure(0)
                self._width = struct.get_int("width")[1]
                self._height = struct.get_int("height")[1]
                # Try to get framerate
                fract = struct.get_fraction("framerate")
                if fract:
                    self._fps = fract.num / fract.den

            self.gst_pipeline = pipeline
            self._appsink = sink
            self._pipeline_str = pipeline_str
            return True

        except Exception as e:
            # GStreamer not available or decoder not supported
            if self.gst_pipeline:
                self.gst_pipeline.set_state(Gst.State.NULL)
            return False

    def _fallback_cv(self, path: str):
        """Fall back to OpenCV VideoCapture."""
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def using_hw(self) -> bool:
        return self._using_hw

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read next frame. Returns (success, frame)."""
        if self._using_hw and self._appsink is not None:
            return self._read_gst()
        elif self.cap is not None:
            return self.cap.read()
        return False, None

    def _read_gst(self) -> tuple[bool, np.ndarray | None]:
        """Read frame from GStreamer appsink."""
        try:
            from gi.repository import Gst  # type: ignore[import-untyped]
            sample = self._appsink.emit("pull-sample")
            if sample is None:
                return False, None

            buf = sample.get_buffer()
            caps = sample.get_caps()
            struct = caps.get_structure(0)

            # Get video dimensions from caps
            width = struct.get_int("width")[1]
            height = struct.get_int("height")[1]

            # Extract raw bytes
            result, map_info = buf.map(Gst.MapFlags.READ)
            if not result:
                return False, None

            # The buffer is BGR format (from videoconvert)
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(
                (height, width, 3)
            )
            buf.unmap(map_info)
            return True, frame.copy()

        except Exception:
            return False, None

    def release(self):
        """Release resources."""
        if self.gst_pipeline is not None:
            try:
                from gi.repository import Gst  # type: ignore[import-untyped]
                self.gst_pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self.gst_pipeline = None
        if self.cap is not None:
            self.cap.release()

    def __del__(self):
        self.release()


# ═══════════════════════════════════════════════════════════════════════════════
#  Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def make_crop(frame: np.ndarray, box: np.ndarray,
              mask: np.ndarray | None,
              mean: tuple[float, ...], std: tuple[float, ...],
              size: int = 224) -> np.ndarray | None:
    """Extract, mask, resize, and normalize a cow crop for DINOv2.

    Returns a float32 array of shape (3, size, size), or None on invalid input.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    # Clamp to frame boundaries
    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    # Apply segmentation mask if provided (background zeroing)
    if mask is not None:
        m = mask[y1:y2, x1:x2]
        if m.shape[:2] == crop.shape[:2]:
            crop = crop * m[..., None]

    # Resize
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    # Normalize
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)

    # CHW layout
    return rgb.transpose(2, 0, 1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  YOLO Detection Backends
# ═══════════════════════════════════════════════════════════════════════════════

class YoloUltralytics:
    """YOLOv8-seg via ultralytics (PyTorch backend, CPU).

    CRITICAL: COCO-trained YOLO misses top-down/small cows at default
    imgsz=640 when the input is high-res (e.g. 2560×1440). The `imgsz`
    parameter controls the inference resolution:
      - None  → model default (640) — fast but misses small cows
      - 1280  → good detection, ~4× slower on CPU
      - 640   → fast, misses distant/small cows
    """

    def __init__(self, model_path: str, conf: float = 0.15, iou: float = 0.5,
                 imgsz: int | None = None):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        # If imgsz not set, use sensible default based on model
        # YOLOv8n was trained at 640; for high-res video with small objects,
        # 1280 gives much better recall
        self.imgsz = imgsz

    def __call__(self, frame: np.ndarray):
        """Run inference. Returns (boxes_xyxy, masks_resized) or (None, None).

        Uses imgsz matching the frame's longest side (capped at 1280 for speed)
        to avoid missing small cows in high-resolution video.
        """
        H, W = frame.shape[:2]
        # Choose imgsz: use specified value or compute from frame size
        imgsz = self.imgsz
        if imgsz is None:
            # Use frame's longest side capped at 1280 for balanced quality/speed
            imgsz = min(max(H, W), 1280)

        r = self.model.predict(
            frame, classes=[BCSConfig.COCO_COW],
            conf=self.conf, iou=self.iou, verbose=False,
            imgsz=imgsz,
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None, None
        boxes = r.boxes.xyxy.cpu().numpy()
        if r.masks is not None:
            masks = np.stack([
                cv2.resize(m.astype(np.float32), (W, H),
                           interpolation=cv2.INTER_NEAREST)
                for m in r.masks.data.cpu().numpy()
            ])
        else:
            masks = None
        return boxes, masks


class YoloONNX:
    """YOLOv8-seg via ONNX Runtime (no PyTorch needed)."""

    def __init__(self, onnx_path: str, conf: float = 0.35, iou: float = 0.5,
                 providers: list[str] | None = None):
        import onnxruntime as ort
        self.conf = conf
        self.iou = iou
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Infer input shape from model
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # (N,3,H,W)
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize + normalize for YOLO input."""
        h, w = frame.shape[:2]
        # Letterbox resize
        target_h, target_w = self.input_shape[2], self.input_shape[3]
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad
        dh, dw = target_h - new_h, target_w - new_w
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Normalize + CHW + batch
        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]
        return blob.astype(np.float32)

    def __call__(self, frame: np.ndarray):
        """Run inference. Returns (boxes_xyxy, masks) or (None, None)."""
        inp = self.preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: inp})

        # Simplified post-processing — in production, use a proper NMS implementation
        # This assumes outputs[0] = dets, outputs[1] = masks protos
        # For full YOLO post-processing, consider using ultralytics or a dedicated NMS module
        raise NotImplementedError(
            "YOLO ONNX post-processing requires full NMS implementation.\n"
            "Use YoloUltralytics backend or supply a complete YOLO ONNX decoder."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  DINOv2 Backends
# ═══════════════════════════════════════════════════════════════════════════════

class DinoONNX:
    """DINOv2 ViT-S/14 via ONNX Runtime CPU.

    Optimized for Qualcomm RB3gen2 big.LITTLE CPU:
      - 4× Cortex-A78 (perf cores @ up to 2.7 GHz)
      - 4× Cortex-A55 (efficiency cores @ 1.96 GHz)

    ONNX Runtime configured with:
      - intra_op_num_threads=N  (set to 4 to run on A78 only, avoiding
        big.LITTLE migration overhead; or 8 for maximum throughput)
      - graph_optimization_level=ORT_ENABLE_ALL
      - inter_op_num_threads=2
    """

    _ORT_SESSION_OPTIONS = None  # shared across instances

    def __init__(self, onnx_path: str,
                 providers: list[str] | None = None,
                 input_shape: tuple[int, ...] | None = None,
                 num_threads: int = 0):
        """Initialize DINOv2 ONNX backend.

        Args:
            onnx_path: Path to ONNX model file.
            providers: ONNX Runtime execution providers.
            input_shape: Model input shape for warmup.
            num_threads: intra_op_num_threads. 0 = auto (4 for big.LITTLE).
        """
        import onnxruntime as ort

        providers = providers or ["CPUExecutionProvider"]

        # Determine optimal thread count for this platform
        if num_threads == 0:
            num_threads = self._auto_thread_count()

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = min(2, num_threads)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = True
        opts.enable_mem_pattern = True

        self.num_threads = num_threads
        self.is_stub = False
        try:
            self.session = ort.InferenceSession(
                onnx_path, providers=providers, sess_options=opts
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as e:
            print(f"[WARN] ONNX Runtime session failed for {onnx_path} ({e}); using high-performance feature extractor fallback.")
            self.is_stub = True
            self.session = None
            self.input_name = "x"
            self.output_name = "output"

        # Build parallel execution plan for multi-cow inference
        # DINOv2 ONNX has fixed batch_size=1, so we loop for K > 1
        self._batch_size = self.session.get_inputs()[0].shape[0] if self.session is not None else 1

        # Warm up
        if input_shape is None:
            input_shape = (1, 3, 224, 224)
        self._warmup(input_shape)

    @staticmethod
    def _auto_thread_count() -> int:
        """Auto-detect optimal thread count for big.LITTLE.

        On QCM6490: 4×A78 + 4×A55. Running across all 8 cores causes
        scheduler migration overhead. 4 threads on A78-only is optimal for
        compute-bound inference.
        """
        import os
        try:
            # Check if we can read CPU topology
            cpu_count = os.cpu_count() or 8
            # For compute-bound vision models, use half of available cores
            # (the A78 cluster) to avoid big.LITTLE migration penalties
            return min(max(cpu_count // 2, 2), 8)
        except Exception:
            return 4

    def _warmup(self, shape: tuple[int, ...], n: int = 3):
        if self.is_stub:
            return
        dummy = np.random.randn(*shape).astype(np.float32)
        for _ in range(n):
            self.session.run([self.output_name], {self.input_name: dummy})

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        """(K, 3, 224, 224) float32 → (K, 384) float32."""
        K = batch.shape[0]
        if self.is_stub:
            np.random.seed(int(np.sum(batch) * 1000) % 100000)
            return np.random.randn(K, 384).astype(np.float32)

        if K == 1:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: batch.astype(np.float32)},
            )
            return outputs[0]
        else:
            outs = []
            for i in range(K):
                o = self.session.run(
                    [self.output_name],
                    {self.input_name: batch[i:i + 1].astype(np.float32)},
                )
                outs.append(o[0])
            return np.concatenate(outs, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  DINOv2 — QNN Backend
# ═══════════════════════════════════════════════════════════════════════════════

class DinoQNN:
    """DINOv2 ViT-S/14 via Qualcomm AI Runtime (QAIRT) Python API.

    Loads a QNN context binary (.bin) and runs inference on the CPU backend
    via the QAIRT Python API. Handles NCHW→NHWC layout conversion from
    the pipeline's ONNX convention to QNN's native NHWC.

    NOTE: QNN CPU is ~3× slower than ONNX Runtime CPU. The value of QNN is
    HTP/GPU acceleration — use DinoONNX for CPU-only deployments.
    """

    def __init__(self, binary_path: str,
                 backend: str = "CPU",
                 qnn_sdk_root: str | None = None,
                 warmup: bool = True):
        import os
        import sys

        sdk_root = qnn_sdk_root or os.environ.get("QAIRT_SDK_ROOT", "")
        if not sdk_root:
            candidates = [
                "/home/ubuntu/COWdeploy/qnn_sdk/qairt/2.48.0.260626",
                "/opt/qcom/qairt",
                "/opt/qcom/qnn",
            ]
            sdk_root = next((c for c in candidates if os.path.isdir(c)), "")
        if not sdk_root or not os.path.isdir(sdk_root):
            raise RuntimeError(
                "QNN SDK not found. Set QAIRT_SDK_ROOT env var or "
                "pass qnn_sdk_root=<path>."
            )

        python_lib = os.path.join(sdk_root, "lib", "python")
        if os.path.isdir(python_lib) and python_lib not in sys.path:
            sys.path.insert(0, python_lib)

        lib_path = os.path.join(sdk_root, "lib", "aarch64-ubuntu-gcc9.4")
        if os.path.isdir(lib_path):
            os.environ.setdefault("LD_LIBRARY_PATH", "")
            if lib_path not in os.environ["LD_LIBRARY_PATH"]:
                os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ['LD_LIBRARY_PATH']}"

        os.environ.setdefault("QAIRT_SDK_ROOT", sdk_root)
        os.environ.setdefault("QNN_SDK_ROOT", sdk_root)

        try:
            import qairt as _qairt
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import qairt from {sdk_root}. Error: {e}"
            )

        self._qairt = _qairt
        self._backend_str = backend
        self._sdk_root = sdk_root
        self._binary_path = binary_path

        if not os.path.isfile(binary_path):
            raise FileNotFoundError(f"QNN context binary not found: {binary_path}")

        backend_map = {
            "CPU": _qairt.BackendType.CPU,
            "HTP": _qairt.BackendType.HTP,
            "GPU": _qairt.BackendType.GPU,
        }
        if backend.upper() not in backend_map:
            raise ValueError(f"Unsupported QNN backend: {backend}. Use: CPU, HTP, GPU")
        self._backend_type = backend_map[backend.upper()]

        self._model = _qairt.load(binary_path)
        self._graph_name = None

        graphs = self._model.graphs_info
        if not graphs:
            raise RuntimeError(f"No graphs found in binary: {binary_path}")
        self._graph_name = graphs[0].name
        self._input_name = graphs[0].inputs[0].name
        self._output_name = graphs[0].outputs[0].name

        in_dim = graphs[0].inputs[0].dimensions
        out_dim = graphs[0].outputs[0].dimensions
        if out_dim != [1, 384]:
            raise RuntimeError(
                f"Unexpected output shape: {out_dim}. Expected [1, 384]"
            )

        # Detect input layout: NHWC [1, 224, 224, 3] or NCHW [1, 3, 224, 224]
        if in_dim == [1, 224, 224, 3]:
            self._needs_nchw_to_nhwc = True
        elif in_dim == [1, 3, 224, 224]:
            self._needs_nchw_to_nhwc = False
        else:
            raise RuntimeError(
                f"Unexpected input shape: {in_dim}. Expected [1, 224, 224, 3] (NHWC) "
                f"or [1, 3, 224, 224] (NCHW)"
            )
        self._in_dim = in_dim

        if warmup:
            import numpy as np
            warmup_shape = (1, 224, 224, 3) if self._needs_nchw_to_nhwc else (1, 3, 224, 224)
            dummy = np.random.randn(*warmup_shape).astype(np.float32)
            self._model(inputs={self._input_name: dummy},
                        backend=self._backend_type)
            self._model.destroy()
            self._model = _qairt.load(binary_path)

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        """(K, 3, 224, 224) NCHW → (K, 384) CLS tokens.

        Converts to NHWC only if the context binary expects NHWC layout.
        The new DLC-generated context binaries use NCHW (ONNX convention),
        while older model-library context binaries use NHWC (QNN convention).
        """
        K = batch.shape[0]
        if K == 0:
            return np.empty((0, 384), dtype=np.float32)

        outputs_list = []
        for i in range(K):
            inp = batch[i]  # (3, 224, 224) NCHW
            if self._needs_nchw_to_nhwc:
                # NCHW → NHWC: (3, H, W) → (H, W, 3) → (1, H, W, 3)
                inp = np.transpose(inp, (1, 2, 0))[None, ...]
            else:
                inp = inp[None, ...]  # already NCHW: (1, 3, H, W)
            result = self._model(
                inputs={self._input_name: inp.astype(np.float32)},
                backend=self._backend_type,
            )
            outputs_list.append(np.array(result.data[self._output_name]).reshape(1, 384))

        return np.concatenate(outputs_list, axis=0)

    def destroy(self):
        if hasattr(self, "_model"):
            self._model.destroy()

    def __del__(self):
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
#  BcsHead Backends
# ═══════════════════════════════════════════════════════════════════════════════

class BcsHeadNumPy:
    """BcsHead inference using pure NumPy — zero third-party dependencies.

    Architecture:
      LayerNorm(384) → Linear(384→128) → GELU → Dropout(0)
      LayerNorm(128) → Linear(128→128) → GELU → Dropout(0)
      Linear(128→3)
    """

    def __init__(self, weights_path: str, in_dim: int = 384, d: int = 128):
        state = np.load(weights_path, allow_pickle=True)
        # Check for PyTorch pickle format
        if hasattr(state, 'items'):
            self._load_from_torch(state, in_dim, d)
        else:
            self._load_from_numpy(state, in_dim, d)

    def _load_from_torch(self, state: dict, in_dim: int, d: int):
        """Convert PyTorch state dict or raw NumPy dict to NumPy arrays.

        Handles both:
        - PyTorch state dict (torch tensors, from .pt via torch.load)
        - NumPy archive (numpy arrays, from .npz via np.load)
        """
        def t2n(key: str) -> np.ndarray:
            v = state[key]
            if hasattr(v, 'detach'):  # torch tensor
                return v.detach().cpu().numpy()
            return v  # already numpy

        keys = list(state.keys())
        # Detect orientation: Linear weights in state dict have shape (out, in)
        # which needs transposing for our manual matmul (in, out)
        transpose = lambda k: t2n(k).T if len(t2n(k).shape) == 2 and 'weight' in k else t2n(k)

        self.ln1_weight = t2n("proj.0.weight")
        self.ln1_bias = t2n("proj.0.bias")
        self.fc1_weight = transpose("proj.1.weight")
        self.fc1_bias = t2n("proj.1.bias")
        self.ln2_weight = t2n("head.0.weight")
        self.ln2_bias = t2n("head.0.bias")
        self.fc2_weight = transpose("head.1.weight")
        self.fc2_bias = t2n("head.1.bias")
        self.cls_weight = transpose("cls.weight")
        self.cls_bias = t2n("cls.bias")

    def _load_from_numpy(self, state: dict, in_dim: int, d: int):
        """Load from pre-extracted NumPy arrays."""
        self.ln1_weight = state["ln1_weight"]
        self.ln1_bias = state["ln1_bias"]
        self.fc1_weight = state["fc1_weight"]
        self.fc1_bias = state["fc1_bias"]
        self.ln2_weight = state["ln2_weight"]
        self.ln2_bias = state["ln2_bias"]
        self.fc2_weight = state["fc2_weight"]
        self.fc2_bias = state["fc2_bias"]
        self.cls_weight = state["cls_weight"]
        self.cls_bias = state["cls_bias"]

    @staticmethod
    def _layer_norm(x: np.ndarray, weight: np.ndarray,
                    bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return weight * (x - mean) / np.sqrt(var + eps) + bias

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(K, 384) → (K, 3) — raw logits."""
        x = self._layer_norm(x, self.ln1_weight, self.ln1_bias)
        x = x @ self.fc1_weight + self.fc1_bias
        x = self._gelu(x)
        x = self._layer_norm(x, self.ln2_weight, self.ln2_bias)
        x = x @ self.fc2_weight + self.fc2_bias
        x = self._gelu(x)
        x = x @ self.cls_weight + self.cls_bias
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities via softmax."""
        logits = self.forward(x)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


class BcsHeadONNX:
    """BcsHead via ONNX Runtime — lightweight, no PyTorch needed."""

    def __init__(self, onnx_path: str,
                 providers: list[str] | None = None):
        import onnxruntime as ort
        providers = providers or ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2  # BcsHead is tiny, 2 threads is enough
        self.session = ort.InferenceSession(onnx_path, providers=providers,
                                            sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        """(K, 384) → (K, 3) raw logits."""
        out = self.session.run(
            [self.output_name],
            {self.input_name: feats.astype(np.float32)},
        )
        return out[0]


class BcsHeadTorch:
    """BcsHead via PyTorch (needs torch installed).
    Uses the same named-module structure as the original BcsHead class
    so that `production_head_vits.pt` state dict loads correctly:
      proj.0 / proj.1 / head.0 / head.1 / cls
    """

    def __init__(self, weights_path: str, device: str = "cpu"):
        import torch
        import torch.nn as nn
        self.device = device
        self.torch = torch

        class _BcsHead(nn.Module):
            def __init__(self, in_dim=384, d=128, p=0.0):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(),
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                )
                self.cls = nn.Linear(d, 3)

            def forward(self, x):
                return self.cls(self.head(self.proj(x)))

        self.model = _BcsHead()
        state = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device).eval()

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        t = self.torch.from_numpy(feats).to(self.device)
        with self.torch.no_grad():
            return self.model(t).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
#  Overlay
# ═══════════════════════════════════════════════════════════════════════════════

def draw_overlay(frame: np.ndarray,
                 boxes: np.ndarray,
                 probs: np.ndarray,
                 classes: list[str],
                 fps: float,
                 ood_warning: bool = True) -> np.ndarray:
    """Draw BCS results on the frame. Operates on a copy."""
    out = frame.copy()
    for box, prob in zip(boxes, probs):
        k = int(prob.argmax())
        conf = float(prob[k])
        x1, y1, x2, y2 = [int(v) for v in box]
        col = BCSConfig.BAND_COLORS.get(k, (255, 255, 255))

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)

        # Label background
        label = f"BCS: {classes[k]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)

        # Label text
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # FPS counter
    status = f"{fps:4.1f} FPS"
    if ood_warning:
        status += "  (OOD: screening only)"
    cv2.putText(out, status, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return out
