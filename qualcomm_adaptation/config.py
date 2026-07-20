"""
BCS Configuration — Qualcomm RB3gen2 adapted.
Mirrors production_config.json with runtime defaults and platform-specific overrides.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class BCSConfig:
    """Unified config for the BCS pipeline on Qualcomm platforms."""

    # ── classes ──────────────────────────────────────────────────────────────
    classes: list[str] = field(default_factory=lambda: ["thin", "ideal", "fat"])
    n_classes: int = 3

    # ── preprocessing ────────────────────────────────────────────────────────
    resize: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ── model paths (set at runtime) ─────────────────────────────────────────
    yolo_path: str = ""
    dino_onnx_path: str = ""
    head_path: str = ""

    # ── inference knobs ──────────────────────────────────────────────────────
    yolo_conf: float = 0.15  # COCO-trained YOLO misses top-down/small cows at 0.35; 0.15 catches most
    yolo_iou: float = 0.50
    frame_skip: int = 0          # 0 = process every frame, 1 = every 2nd, etc.
    max_frames: int = 0          # 0 = all frames
    input_scale: float = 1.0     # <1.0 downscales the input video (e.g. 0.5 = 1280×720)
    dino_providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])

    # ── runtime flags ────────────────────────────────────────────────────────
    use_onnx_yolo: bool = False  # ultralytics is used by default; True = raw ONNX YOLO
    no_display: bool = True      # headless (no cv2.imshow)
    benchmark: bool = False      # run in benchmark mode (no overlay drawing)
    hw_decode: bool = False      # use GStreamer V4L2 HW-accelerated video decode
    num_threads: int = 0         # ONNX Runtime intra_op_num_threads (0 = auto)

    # ── constants ────────────────────────────────────────────────────────────
    COCO_COW: ClassVar[int] = 19

    # BGR colors for each band
    BAND_COLORS: ClassVar[dict[int, tuple[int, int, int]]] = {
        0: (60, 76, 231),    # thin → blue
        1: (104, 168, 85),   # ideal → green
        2: (82, 78, 196),    # fat → red
    }

    # ── model architecture ───────────────────────────────────────────────────
    dino_in_dim: int = 384
    head_d_model: int = 128
    head_dropout: float = 0.0  # no dropout at inference
    dino_input_shape: tuple[int, int, int] = (3, 224, 224)

    @classmethod
    def from_json(cls, path: str, **overrides) -> "BCSConfig":
        """Load from a JSON config file and apply any keyword overrides."""
        raw = json.loads(Path(path).read_text())
        cfg = cls()

        if "classes" in raw:
            cfg.classes = raw["classes"]
            cfg.n_classes = len(cfg.classes)

        if "preprocess" in raw:
            p = raw["preprocess"]
            cfg.resize = p.get("resize", cfg.resize)
            cfg.mean = tuple(p.get("mean", cfg.mean))
            cfg.std = tuple(p.get("std", cfg.std))

        if "in_dim" in raw:
            cfg.dino_in_dim = raw["in_dim"]
        if "d_model" in raw:
            cfg.head_d_model = raw["d_model"]

        # Apply runtime overrides
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        return cfg

    def __post_init__(self):
        assert len(self.classes) == self.n_classes, (
            f"Expected {self.n_classes} classes, got {len(self.classes)}"
        )
        assert len(self.mean) == 3 and len(self.std) == 3
