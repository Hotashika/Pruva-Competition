from njord.config.camera_config import CAMERA_WIDTH, DEPTH_SHAPE, RGB_SHAPE
from njord.config.vision_config import (
    AR_TAG_MODEL_PATH,
    BUOY_MODEL_PATH,
    DEVICE,
    TOLERANCE_DEG,
    TOLERANCE_RATIO,
)
from njord.core import shared_state
from njord.core.shared_memory_utils import (
    attach_existing_shared_memory,
    close_shared_memory_handles,
)
from vision.detector import ArTagDetector, BuoyDetector

TASK_DETECTOR_MAP = {
    "task1": {"buoy"},
    "task2": {"buoy"},
    "task3": {"ar_tag"},
    "task4": {"buoy"},
}

DETECTOR_SPECS = {
    "buoy": {
        "class": BuoyDetector,
        "model_path": BUOY_MODEL_PATH,
    },
    "ar_tag": {
        "class": ArTagDetector,
        "model_path": AR_TAG_MODEL_PATH,
    },
}

QR_TOPIC = "/njord/task3/qr_detections"
QR_TASK = "task3"
AR_CONFIRMED_HZ_ENV = "NJORD_TASK3_AR_CONFIRMED_HZ"
USE_SHARED_CALIBRATION = True
CALIBRATION_FX_INDEX = 0
CALIBRATION_CX_INDEX = 2
PUBLISH_CAMERA_TIMESTAMP = True
