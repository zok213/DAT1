#!/usr/bin/env python3
"""
Convert BcsHead from PyTorch .pt to ONNX for deployment.
Also exports the weights as pure NumPy arrays for the numpy backend.

Usage:
  python3 scripts/convert_head_to_onnx.py \
      --input production_head_vits.pt \
      --output models/bcs_head.onnx \
      --config production_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Convert BcsHead to ONNX + NumPy")
    ap.add_argument("--input", required=True, help="PyTorch .pt weights")
    ap.add_argument("--output-onnx", default=None, help="Output .onnx path")
    ap.add_argument("--output-numpy", default=None, help="Output .npz path (NumPy weights)")
    ap.add_argument("--config", default="production_config.json", help="Config JSON")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    in_dim = cfg.get("in_dim", 384)
    d_model = cfg.get("d_model", 128)

    # Default outputs
    onnx_path = args.output_onnx or Path(args.input).with_suffix(".onnx").name
    npz_path = args.output_numpy or Path(args.input).with_suffix(".npz").name

    # ── Load PyTorch weights ─────────────────────────────────────────────
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
        print("[warn] PyTorch not installed; extracting from saved state dict may fail")

    if has_torch:
        state = torch.load(args.input, map_location="cpu")
        if isinstance(state, dict):
            pass  # good: it's a state dict
        elif hasattr(state, "state_dict"):
            state = state.state_dict()
        else:
            state = state  # assume it's a module or compatible type

        # Print keys for verification
        print(f"[info] Loaded state dict with {len(state)} keys:")
        for k, v in state.items():
            print(f"  {k}: {tuple(v.shape)}")

        # ── Build and export to ONNX ─────────────────────────────────────
        import torch.nn as nn

        class BcsHeadONNX(nn.Module):
            """BcsHead with dropout removed for inference."""
            def __init__(self, in_dim=384, d=128):
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

        model = BcsHeadONNX(in_dim, d_model)
        model.load_state_dict(state)
        model.eval()

        # Export
        dummy = torch.randn(1, in_dim)
        torch.onnx.export(
            model, dummy,
            onnx_path,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={
                "features": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"[ok] ONNX exported → {onnx_path}")

        # ── Export NumPy weights ─────────────────────────────────────────
        np_weights = {}
        for k, v in state.items():
            np_weights[k] = v.detach().cpu().numpy()
        np.savez_compressed(npz_path, **np_weights)
        print(f"[ok] NumPy weights exported → {npz_path}")

        # ── Validate ONNX ────────────────────────────────────────────────
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"[ok] ONNX validation passed")
        except Exception as e:
            print(f"[warn] ONNX validation: {e}")

        # ── Validate inference ───────────────────────────────────────────
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            test_input = np.random.randn(4, in_dim).astype(np.float32)
            output = session.run(None, {"features": test_input})[0]
            print(f"[ok] ONNX inference test: input {test_input.shape} → output {output.shape}")
            print(f"     Sample logits: {output[0]}")
        except Exception as e:
            print(f"[warn] ONNX runtime test: {e}")

    else:
        print("[err] PyTorch is required for model conversion")
        print("      Install with: pip install torch --index-url https://piwheels.org/simple")
        sys.exit(1)

    print(f"\n[done] Outputs:")
    print(f"  ONNX:   {onnx_path}")
    print(f"  NumPy:  {npz_path}")


if __name__ == "__main__":
    main()
