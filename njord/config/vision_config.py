from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent

# Active Njord object detectors. Vessel detection is intentionally not used;
# every mission consumes detections from the buoy and AR-tag models.
BUOY_MODEL_PATH = str(BASE_DIR / "models" / "buoy" / "buoy.pt")
AR_TAG_MODEL_PATH = str(BASE_DIR / "models" / "ar_tag" / "ar_tag.pt")

TOLERANCE_RATIO = 0.05  # Tolerance ratio for bounding box size filtering
TOLERANCE_DEG = 5  # Tolerance deg for tolerance ratio

# Device selection for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
