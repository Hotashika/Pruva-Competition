import threading

latest_frame = None
latest_frame_id = 0
frame_lock = threading.Lock()
frame_condition = threading.Condition(frame_lock)

latest_center_depth = None  # nearest detection bbox median, metres
latest_imu = None  # roll, pitch, yaw
latest_timestamp = None  # ms
latest_data_id = 0
data_lock = threading.Lock()
data_condition = threading.Condition(data_lock)

RGB_SHM_NAME = "RGB_DATA"
DEPTH_SHM_NAME = "DEPTH_DATA"
DEPTH_VISION_SHM_NAME = "DEPTH_VISION"
META_SHM_NAME = "ZED_META"
IMU_SHM_NAME = "ZED_IMU"
CALIB_SHM_NAME = "ZED_CALIB"

META_SHAPE = (2,)  # frame_id, image timestamp in ms
IMU_SHAPE = (3,)  # roll, pitch, yaw in radians
CALIB_SHAPE = (4,)  # fx, fy, cx, cy
