from teknofest.config.camera_config import CAMERA_WIDTH, DEPTH_SHAPE, RGB_SHAPE
from teknofest.config.vision_config import (
    BUOY_MODEL_PATH,
    DEVICE,
    TOLERANCE_DEG,
    TOLERANCE_RATIO,
)
from teknofest.core import shared_state
from teknofest.core.shared_memory_utils import (
    attach_existing_shared_memory,
    close_shared_memory_handles,
)
from vision.detector import BuoyDetector

TASK_DETECTOR_MAP = {
    "task1": {"buoy"},
    "task2": {"buoy"},
    "task3": {"buoy"},
}

DETECTOR_SPECS = {
    "buoy": {
        "class": BuoyDetector,
        "model_path": BUOY_MODEL_PATH,
    },
}

STARTUP_DETECTORS = ("buoy",)

QR_TOPIC = None
QR_TASK = None
AR_CONFIRMED_HZ_ENV = None
USE_SHARED_CALIBRATION = False
CALIBRATION_FX_INDEX = None
CALIBRATION_CX_INDEX = None
PUBLISH_CAMERA_TIMESTAMP = False
