import math
import os
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Active Njord object detectors. Vessel detection is intentionally not used;
# every mission consumes detections from the buoy and AR-tag models.

BUOY_MODEL_PATH = str(REPOSITORY_ROOT / "models" / "njord_buoy" / "njord_buoy.engine")
AR_TAG_MODEL_PATH = str(REPOSITORY_ROOT / "models" / "ar_tag" / "ar_tag.engine")
EWASR_MODEL_PATH = os.getenv(
    "NJORD_EWASR_MODEL_PATH",
    str(
        REPOSITORY_ROOT
        / "models"
        / "ewasr"
        / "ewasr_resnet18_imu.torchscript"
    ),
)


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_positive_float(name, default):
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) and value > 0.0 else float(default)


EWASR_ENABLED = env_flag("NJORD_EWASR_ENABLED", True)
TASK2_FUSION_SHADOW_MODE = env_flag(
    "NJORD_TASK2_FUSION_SHADOW",
    True,
)
EWASR_INFERENCE_HZ = env_positive_float(
    "NJORD_EWASR_INFERENCE_HZ",
    5.0,
)

TOLERANCE_RATIO = 0.05  # Tolerance ratio for bounding box size filtering
TOLERANCE_DEG = 5  # Tolerance deg for tolerance ratio

# Device selection for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
