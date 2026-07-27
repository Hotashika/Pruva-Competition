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
from vision.usv_3d_detector import USV3DObstacleDetector

TASK_DETECTOR_MAP = {
    "task1": {"buoy"},
    "task2": {"usv_3d"},
    "task3": {"ar_tag"},
    "task4": {"buoy"},
}

DETECTOR_SPECS = {
    "buoy": {
        "class": BuoyDetector,
        "model_path": BUOY_MODEL_PATH,
    },
    "usv_3d": {
        "class": USV3DObstacleDetector,
        "uses_full_intrinsics": True,
        "kwargs": {
            "camera_height_m": 0.25,
            "detection_max_range_m": 12.0,
            "plane_ransac_iterations": 350,
        },
    },
    "ar_tag": {
        "class": ArTagDetector,
        "model_path": AR_TAG_MODEL_PATH,
    },
}

STARTUP_DETECTORS = ("buoy", "usv_3d", "ar_tag")

QR_TOPIC = "/njord/task3/qr_detections"
QR_TASK = "task3"
AR_CONFIRMED_HZ_ENV = "NJORD_TASK3_AR_CONFIRMED_HZ"
USE_SHARED_CALIBRATION = True
CALIBRATION_FX_INDEX = 0
CALIBRATION_FY_INDEX = 1
CALIBRATION_CX_INDEX = 2
CALIBRATION_CY_INDEX = 3
PUBLISH_CAMERA_TIMESTAMP = True
