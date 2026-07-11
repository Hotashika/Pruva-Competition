"""
ZED Camera Configuration
"""

import pyzed.sl as sl

# -----------------------------------------------------------------------------
# Camera Settings
# -----------------------------------------------------------------------------

# Resolution
CAMERA_RESOLUTION = sl.RESOLUTION.HD720

RESOLUTION_MAP = {
    sl.RESOLUTION.HD2K: (2208, 1242),
    sl.RESOLUTION.HD1200: (1920, 1200),
    sl.RESOLUTION.HD1080: (1920, 1080),
    sl.RESOLUTION.HD720: (1280, 720),
    sl.RESOLUTION.SVGA: (960, 600),
    sl.RESOLUTION.VGA: (672, 376),
}

CAMERA_WIDTH, CAMERA_HEIGHT = RESOLUTION_MAP[CAMERA_RESOLUTION]

RGB_SHAPE = (CAMERA_HEIGHT, CAMERA_WIDTH, 4)
DEPTH_SHAPE = (CAMERA_HEIGHT, CAMERA_WIDTH)

# FPS
CAMERA_FPS = 15

# Depth Mode
DEPTH_MODE = sl.DEPTH_MODE.NEURAL

# Coordinate Units
COORDINATE_UNITS = sl.UNIT.METER

# Coordinate System
COORDINATE_SYSTEM = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

# -----------------------------------------------------------------------------
# Runtime Settings
# -----------------------------------------------------------------------------

# Fill mode for depth map
ENABLE_FILL_MODE = True

# Confidence threshold (0-100)
CONFIDENCE_THRESHOLD = 50

# Texture confidence threshold (0-100)
TEXTURE_CONFIDENCE_THRESHOLD = 50
