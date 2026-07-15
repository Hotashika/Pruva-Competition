import threading

latest_frame = None
frame_lock = threading.Lock()
frame_event = threading.Event()

latest_depth_array = None  # np array
latest_imu = None  # roll, pitch, yaw
latest_timestamp = None  # ms
data_lock = threading.Lock()
data_event = threading.Event()

RGB_SHM_NAME = "RGB_DATA"
DEPTH_SHM_NAME = "DEPTH_DATA"
META_SHM_NAME = "ZED_META"
IMU_SHM_NAME = "ZED_IMU"
CALIB_SHM_NAME = "ZED_CALIB"

META_SHAPE = (2,)  # frame_id, image timestamp in ms
IMU_SHAPE = (3,)  # roll, pitch, yaw in radians
CALIB_SHAPE = (4,)  # fx, fy, cx, cy
