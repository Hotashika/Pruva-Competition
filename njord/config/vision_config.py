from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths for PyTorch
BUOY_MODEL_PATH = str(BASE_DIR / "models" / "buoy" / "buoy.engine")
VESSEL_MODEL_PATH = str(BASE_DIR / "models" / "vessel" / "vessel.engine")
HUMAN_MODEL_PATH = str(BASE_DIR / "models" / "human" / "human.engine")

TOLERANCE_RATIO = 0.05  # Tolerance ratio for bounding box size filtering
TOLARANCE_DEG = 3  # Tolerance deg for tolerance ratio

# Device selection for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
