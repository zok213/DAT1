"""
Qualcomm QNN Backend — hardware-accelerated inference via QAIRT Python API.

Provides drop-in replacements for ONNX Runtime / PyTorch backends using
the Qualcomm AI Runtime (QAIRT) Python API. Supports CPU, HTP (Hexagon DSP),
and GPU (Adreno) backends.

The QAIRT Python API wraps the native QNN C++ runtime. It is the recommended
interface for programmatic inference, as opposed to the lower-level qnn-net-run
CLI or the raw QNN C API.

Usage:
    from qualcomm_adaptation.qnn_backend import DinoQNN

    dino = DinoQNN("models/dinov2_fp32_cpu.bin.bin", backend="CPU")
    feats = dino(crops)  # (K, 384)

Backend Availability:
    - CPU:  ✅ Works (but ONNX Runtime is 3× faster on CPU)
    - HTP:  ❌ Blocked — requires libcdsprpc.so (Qualcomm CDSP userspace driver)
    - GPU:  ❌ Blocked — requires Adreno compute driver (kgsl kernel module)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  QAIRT SDK detection
# ═══════════════════════════════════════════════════════════════════════════════

QAIRT_AVAILABLE = False
QAIRT_SDK_ROOT: str | None = None

for _root in [
    os.environ.get("QAIRT_SDK_ROOT", ""),
    os.environ.get("QNN_SDK_ROOT", ""),
    "/home/ubuntu/COWdeploy/qnn_sdk/qairt/2.48.0.260626",
    "/opt/qcom/qairt",
    "/opt/qcom/qnn",
]:
    if _root and Path(_root, "lib", "python").exists():
        QAIRT_SDK_ROOT = _root
        QAIRT_AVAILABLE = True
        break


def _setup_qairt_env(sdk_root: str):
    """Set up environment variables and Python path for QAIRT."""
    python_lib = Path(sdk_root) / "lib" / "python"
    if python_lib.is_dir() and str(python_lib) not in sys.path:
        sys.path.insert(0, str(python_lib))

    # Prefer aarch64-ubuntu libs; fall back to OE-compiled libs
    lib_paths = [
        Path(sdk_root) / "lib" / "aarch64-ubuntu-gcc9.4",
        Path(sdk_root) / "lib" / "linux-aarch64-oe-gcc11.2",
        Path(sdk_root) / "lib",
    ]
    for lp in lib_paths:
        if lp.is_dir():
            os.environ.setdefault("LD_LIBRARY_PATH", "")
            if str(lp) not in os.environ["LD_LIBRARY_PATH"]:
                os.environ["LD_LIBRARY_PATH"] = f"{lp}:{os.environ['LD_LIBRARY_PATH']}"

    os.environ.setdefault("QAIRT_SDK_ROOT", sdk_root)
    os.environ.setdefault("QNN_SDK_ROOT", sdk_root)


def import_qairt():
    """Import the QAIRT Python module. Returns the module or raises."""
    if not QAIRT_SDK_ROOT:
        raise RuntimeError(
            "QAIRT SDK not found. Set QAIRT_SDK_ROOT or install from "
            "https://qpm.qualcomm.com/"
        )
    _setup_qairt_env(QAIRT_SDK_ROOT)
    try:
        import qairt as _m
        return _m
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import qairt from {QAIRT_SDK_ROOT}. Error: {e}\n"
            f"Python path includes: {[p for p in sys.path if 'qairt' in p or 'qnn' in p]}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Drop-in Backends
# ═══════════════════════════════════════════════════════════════════════════════

class DinoQNN:
    """DINOv2 ViT-S/14 via QAIRT Python API — drop-in for DinoONNX.

    Loads a QNN context binary (.bin) and runs inference on the selected backend.

    KNOWN ISSUE: The QAIRT Python API produces slightly different results than
    qnn-net-run for the same context binary (cosine similarity ~0.984 vs ~0.99999).
    This appears to be a bug in the Python native wrapper (libPyNetRun312.so)
    which only has an OE-compiled variant for aarch64, not Ubuntu-compiled.
    For CPU inference, ONNX Runtime is recommended instead.
    """

    _BACKEND_MAP: dict[str, Any] = {}  # lazy: qairt.BackendType.X

    def __init__(self, binary_path: str, backend: str = "CPU"):
        self._qairt = import_qairt()
        self._build_backend_map()

        backend_key = backend.upper()
        if backend_key not in self._BACKEND_MAP:
            raise ValueError(
                f"Unsupported QNN backend: {backend}. Use: {list(self._BACKEND_MAP.keys())}"
            )

        self._binary_path = binary_path
        self._backend_type = self._BACKEND_MAP[backend_key]
        self._backend_str = backend_key

        if not Path(binary_path).is_file():
            raise FileNotFoundError(f"QNN context binary not found: {binary_path}")

        # Load model
        self._model = self._qairt.load(binary_path)

        # Read graph info
        graphs = self._model.graphs_info
        if not graphs:
            raise RuntimeError(f"No graphs found in binary: {binary_path}")
        self._graph_name = graphs[0].name
        self._input_name = graphs[0].inputs[0].name
        self._output_name = graphs[0].outputs[0].name
        self._input_dims = graphs[0].inputs[0].dimensions
        self._output_dims = graphs[0].outputs[0].dimensions

    def _build_backend_map(self):
        if not self._BACKEND_MAP:
            self._BACKEND_MAP = {
                "CPU": self._qairt.BackendType.CPU,
                "HTP": self._qairt.BackendType.HTP,
                "GPU": self._qairt.BackendType.GPU,
            }

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def output_name(self) -> str:
        return self._output_name

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        """(K, 3, 224, 224) NCHW → (K, 384) CLS tokens.

        Converts from the pipeline's ONNX NCHW convention to QNN's native NHWC.
        """
        K = batch.shape[0]
        if K == 0:
            return np.empty((0, 384), dtype=np.float32)

        outputs_list = []
        for i in range(K):
            # NCHW → NHWC for QNN
            nhwc = np.transpose(batch[i], (1, 2, 0))[None, ...]
            result = self._model(
                inputs={self._input_name: nhwc.astype(np.float32)},
                backend=self._backend_type,
            )
            out = np.array(result.data[self._output_name]).reshape(1, 384)
            outputs_list.append(out)

        return np.concatenate(outputs_list, axis=0)

    def destroy(self):
        if hasattr(self, "_model"):
            self._model.destroy()

    def __del__(self):
        self.destroy()


class BcsHeadQNN:
    """BcsHead via QAIRT Python API — drop-in for BcsHeadTorch/ONNX/NumPy.

    Requires a QNN context binary for the BcsHead model.
    The BcsHead is tiny (384→128→128→3) — ONNX Runtime is already near-zero
    latency, so QNN acceleration offers negligible benefit. This class exists
    for completeness when running the full pipeline on QNN.
    """

    _BACKEND_MAP: dict[str, Any] = {}

    def __init__(self, binary_path: str, backend: str = "CPU"):
        self._qairt = import_qairt()
        self._build_backend_map()

        backend_key = backend.upper()
        if backend_key not in self._BACKEND_MAP:
            raise ValueError(
                f"Unsupported QNN backend: {backend}. Use: {list(self._BACKEND_MAP.keys())}"
            )
        self._backend_type = self._BACKEND_MAP[backend_key]

        if not Path(binary_path).is_file():
            raise FileNotFoundError(f"QNN context binary not found: {binary_path}")

        self._model = self._qairt.load(binary_path)
        graphs = self._model.graphs_info
        if not graphs:
            raise RuntimeError(f"No graphs found in binary: {binary_path}")
        self._input_name = graphs[0].inputs[0].name
        self._output_name = graphs[0].outputs[0].name

    def _build_backend_map(self):
        if not self._BACKEND_MAP:
            self._BACKEND_MAP = {
                "CPU": self._qairt.BackendType.CPU,
                "HTP": self._qairt.BackendType.HTP,
                "GPU": self._qairt.BackendType.GPU,
            }

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        """(K, 384) → (K, 3) raw logits."""
        import time
        K = feats.shape[0]
        if K == 0:
            return np.empty((0, 3), dtype=np.float32)

        outputs_list = []
        for i in range(K):
            inp = feats[i : i + 1].astype(np.float32)
            result = self._model(
                inputs={self._input_name: inp},
                backend=self._backend_type,
            )
            outputs_list.append(np.array(result.data[self._output_name]).reshape(1, 3))
        return np.concatenate(outputs_list, axis=0)

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        logits = self(feats)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def destroy(self):
        if hasattr(self, "_model"):
            self._model.destroy()

    def __del__(self):
        self.destroy()


class YoloQNN:
    """YOLOv8-seg via QAIRT Python API — placeholder.

    YOLO QNN backends require full NMS post-processing which is non-trivial
    to implement. Use the ultralytics PyTorch backend for now.

    When QNN SDK with HTP/GPU is available, this class will:
      1. Load a YOLO context binary
      2. Run inference on CDSP/GPU
      3. Apply NMS post-processing

    For now, raises NotImplementedError if instantiated.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "YOLO QNN backend requires NMS post-processing.\n"
            "Use YoloUltralytics (--yolo .pt) for now."
        )

    def __call__(self, frame: np.ndarray):
        raise NotImplementedError(
            "YOLO QNN inference + NMS not yet implemented.\n"
            "Use YoloUltralytics with --yolo .pt"
        )
