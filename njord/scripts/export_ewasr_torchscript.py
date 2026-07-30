#!/usr/bin/env python3
"""Export official eWaSR weights to Njord's stable TorchScript contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "models"
    / "ewasr"
    / "ewasr_resnet18_imu.torchscript"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert official tersekmatija/eWaSR .pth weights into the "
            "two-input TorchScript model consumed by Njord."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Clone of https://github.com/tersekmatija/eWaSR",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Official ewasr_resnet18_imu .pth weights",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--architecture",
        default="ewasr_resnet18_imu",
        choices=("ewasr_resnet18_imu",),
    )
    parser.add_argument("--input-height", type=int, default=384)
    parser.add_argument("--input-width", type=int, default=512)
    return parser


def _load_state_dict(torch, weights_path):
    try:
        payload = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(weights_path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise TypeError("eWaSR weights must contain a PyTorch state dictionary")
    return payload


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(args):
    source_dir = args.source_dir.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not (source_dir / "wasr" / "models.py").is_file():
        raise SystemExit(f"Official eWaSR source not found: {source_dir}")
    if not weights_path.is_file():
        raise SystemExit(f"eWaSR weights not found: {weights_path}")
    if args.input_height <= 0 or args.input_width <= 0:
        raise SystemExit("Input dimensions must be positive")

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for eWaSR export") from exc

    sys.path.insert(0, str(source_dir))
    try:
        import wasr.models as ewasr_models
    except ImportError as exc:
        raise SystemExit(
            "Official eWaSR dependencies are missing. Install its "
            "requirements before exporting."
        ) from exc

    # The official factory requests ImageNet ResNet weights even though the
    # complete released state dict immediately replaces them. Avoid a needless
    # network download so export remains deterministic and offline-friendly.
    original_resnet18 = ewasr_models.resnet18

    def resnet18_without_pretrained_download(*factory_args, **factory_kwargs):
        factory_kwargs.pop("pretrained", None)
        factory_kwargs.pop("weights", None)
        try:
            return original_resnet18(
                *factory_args,
                weights=None,
                **factory_kwargs,
            )
        except TypeError:
            return original_resnet18(
                *factory_args,
                pretrained=False,
                **factory_kwargs,
            )

    ewasr_models.resnet18 = resnet18_without_pretrained_download
    model = ewasr_models.get_model(
        args.architecture,
        num_classes=3,
        pretrained=False,
    )
    model.load_state_dict(_load_state_dict(torch, weights_path))
    model = model.eval().cpu()

    class NjordEWaSRWrapper(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, image, imu_mask):
            return self.wrapped(
                {
                    "image": image,
                    "imu_mask": imu_mask,
                }
            )["out"]

    wrapper = NjordEWaSRWrapper(model).eval()
    example_image = torch.zeros(
        1,
        3,
        args.input_height,
        args.input_width,
    )
    example_imu = torch.zeros(
        1,
        args.input_height,
        args.input_width,
    )
    traced = torch.jit.trace(
        wrapper,
        (example_image, example_imu),
        strict=False,
    )
    with torch.no_grad():
        output = traced(example_image, example_imu)
    if tuple(output.shape[:2]) != (1, 3):
        raise RuntimeError(
            f"Unexpected exported eWaSR output shape: {tuple(output.shape)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    metadata = {
        "architecture": args.architecture,
        "classes": {
            "0": "obstacle",
            "1": "water",
            "2": "sky",
        },
        "input_height": args.input_height,
        "input_width": args.input_width,
        "source_repository": "https://github.com/tersekmatija/eWaSR",
        "source_weights": str(weights_path),
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "torchscript_sha256": _sha256(output_path),
    }
    metadata_path = output_path.with_suffix(
        output_path.suffix + ".json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    print(metadata_path)


def main(argv=None):
    args = build_parser().parse_args(argv)
    export(args)


if __name__ == "__main__":
    main()
