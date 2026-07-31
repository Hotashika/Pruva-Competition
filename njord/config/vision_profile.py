from njord.config.camera_config import CAMERA_WIDTH, DEPTH_SHAPE, RGB_SHAPE
from njord.config.vision_config import (
    AR_TAG_MODEL_PATH,
    BUOY_MODEL_PATH,
    DEVICE,
    EWASR_ENABLED,
    EWASR_INFERENCE_HZ,
    EWASR_MODEL_PATH,
    TASK2_FUSION_SHADOW_MODE,
    TOLERANCE_DEG,
    TOLERANCE_RATIO,
)
from njord.core import shared_state
from njord.core.shared_memory_utils import (
    attach_existing_shared_memory,
    close_shared_memory_handles,
)
from vision.detector import ArTagDetector, BuoyDetector
from vision.task2_fusion_detector import Task2FusionDetector

TASK_DETECTOR_MAP = {
    "task1": {"buoy"},
    # Run the buoy model first so Task 2 can preserve red/yellow semantics
    # before considering unmatched metric-depth clusters as generic obstacles.
    "task2": ("buoy", "task2_fusion"),
    "task3": {"ar_tag"},
    "task4": {"buoy"},
}

DETECTOR_SPECS = {
    "buoy": {
        "class": BuoyDetector,
        "model_path": BUOY_MODEL_PATH,
    },
    "task2_fusion": {
        "class": Task2FusionDetector,
        "model_path": EWASR_MODEL_PATH,
        "uses_full_intrinsics": True,
        "uses_imu": True,
        "kwargs": {
            "camera_height_m": 0.25,
            "detection_max_range_m": 12.0,
            "plane_ransac_iterations": 350,
            "segmentation_enabled": EWASR_ENABLED,
            "shadow_mode": TASK2_FUSION_SHADOW_MODE,
            "segmentation_hz": EWASR_INFERENCE_HZ,
        },
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
USE_SHARED_IMU = True
CALIBRATION_FX_INDEX = 0
CALIBRATION_FY_INDEX = 1
CALIBRATION_CX_INDEX = 2
CALIBRATION_CY_INDEX = 3
PUBLISH_CAMERA_TIMESTAMP = True
SHADOW_DETECTIONS_TOPIC = "/vision/task2_fusion_debug"
