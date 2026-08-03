from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Model paths for PyTorch
BUOY_MODEL_PATH = str(REPOSITORY_ROOT / "models" / "teknofest_buoy" / "teknofest_buoy.engine")


TOLERANCE_RATIO = 0.05  # Tolerance ratio for bounding box size filtering
TOLERANCE_DEG = 3  # Tolerance deg for tolerance ratio

# Device selection for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
