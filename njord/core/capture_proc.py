import signal
from multiprocessing import shared_memory

import numpy as np
import pyzed.sl as sl

from njord.config.camera_config import (
    CAMERA_RESOLUTION,
    CAMERA_FPS,
    DEPTH_MODE,
    COORDINATE_UNITS,
    COORDINATE_SYSTEM,
    RGB_SHAPE,
    DEPTH_SHAPE,
)
from njord.core import shared_state


def _create_owned_shared_memory(name, size):
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        stale_shm = shared_memory.SharedMemory(name=name)
        stale_shm.close()
        stale_shm.unlink()
        return shared_memory.SharedMemory(name=name, create=True, size=size)


# noinspection D
def run_capture(
        rgb_shm_name=shared_state.RGB_SHM_NAME,
        depth_shm_name=shared_state.DEPTH_SHM_NAME,
        depth_vision_shm_name=shared_state.DEPTH_VISION_SHM_NAME,
        meta_shm_name=shared_state.META_SHM_NAME,
        imu_shm_name=shared_state.IMU_SHM_NAME,
        calib_shm_name=shared_state.CALIB_SHM_NAME,
        lock=None,
        frame_ready_event=None,
        stop_event=None,
        ready_queue=None,
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # ------------------------------------------------------------------
    # Open ZED Camera
    # ------------------------------------------------------------------
    zed = sl.Camera()
    rgb_shm = None
    depth_shm = None
    depth_vision_shm = None
    meta_shm = None
    imu_shm = None
    calib_shm = None
    ready_sent = False

    try:
        init = sl.InitParameters()
        init.camera_resolution = CAMERA_RESOLUTION
        init.camera_fps = CAMERA_FPS
        init.depth_mode = DEPTH_MODE
        init.coordinate_units = COORDINATE_UNITS
        init.coordinate_system = COORDINATE_SYSTEM

        status = zed.open(init)

        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")

        cam_info = zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters.left_cam
        fx = float(calib.fx)
        fy = float(calib.fy)
        cx = float(calib.cx)
        cy = float(calib.cy)

        runtime = sl.RuntimeParameters()

        rgb_mat = sl.Mat()
        depth_mat = sl.Mat()
        depth_vision_mat = sl.Mat()
        sensors_data = sl.SensorsData()

        # ------------------------------------------------------------------
        # Create Shared Memory
        # ------------------------------------------------------------------
        rgb_shm = _create_owned_shared_memory(
            rgb_shm_name,
            int(np.prod(RGB_SHAPE) * np.dtype(np.uint8).itemsize),
        )

        depth_shm = _create_owned_shared_memory(
            depth_shm_name,
            int(np.prod(DEPTH_SHAPE) * np.dtype(np.float32).itemsize),
        )

        depth_vision_shm = _create_owned_shared_memory(
            depth_vision_shm_name,
            int(np.prod(RGB_SHAPE) * np.dtype(np.uint8).itemsize),
        )

        meta_shm = _create_owned_shared_memory(
            meta_shm_name,
            int(np.prod(shared_state.META_SHAPE) * np.dtype(np.int64).itemsize),
        )

        imu_shm = _create_owned_shared_memory(
            imu_shm_name,
            int(np.prod(shared_state.IMU_SHAPE) * np.dtype(np.float64).itemsize),
        )

        calib_shm = _create_owned_shared_memory(
            calib_shm_name,
            int(np.prod(shared_state.CALIB_SHAPE) * np.dtype(np.float64).itemsize),
        )

        rgb_buf = np.ndarray(
            RGB_SHAPE,
            dtype=np.uint8,
            buffer=rgb_shm.buf,
        )

        depth_buf = np.ndarray(
            DEPTH_SHAPE,
            dtype=np.float32,
            buffer=depth_shm.buf,
        )

        depth_vision_buf = np.ndarray(
            RGB_SHAPE,
            dtype=np.uint8,
            buffer=depth_vision_shm.buf,
        )

        meta_buf = np.ndarray(
            shared_state.META_SHAPE,
            dtype=np.int64,
            buffer=meta_shm.buf,
        )

        imu_buf = np.ndarray(
            shared_state.IMU_SHAPE,
            dtype=np.float64,
            buffer=imu_shm.buf,
        )

        calib_buf = np.ndarray(
            shared_state.CALIB_SHAPE,
            dtype=np.float64,
            buffer=calib_shm.buf,
        )

        meta_buf[:] = 0
        imu_buf[:] = 0.0
        calib_buf[:] = (fx, fy, cx, cy)
        frame_index = 0

        if ready_queue is not None:
            ready_queue.put({"fx": fx, "fy": fy, "cx": cx, "cy": cy})
            ready_sent = True

        # ------------------------------------------------------------------
        # Capture Loop
        # ------------------------------------------------------------------
        while stop_event is None or not stop_event.is_set():
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(rgb_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            zed.retrieve_image(depth_vision_mat, sl.VIEW.DEPTH)
            zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
            timestamp_ms = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()

            imu_pose = sensors_data.get_imu_data().get_pose()
            try:
                roll, pitch, yaw = imu_pose.get_euler_angles(radian=True)
            except TypeError:
                # Compatibility with ZED Python API releases whose method does
                # not expose the radian keyword. Radians are the SDK default.
                roll, pitch, yaw = imu_pose.get_euler_angles()

            frame_index += 1
            if lock is None:
                rgb_buf[:] = rgb_mat.get_data()
                depth_buf[:] = depth_mat.get_data()
                depth_vision_buf[:] = depth_vision_mat.get_data()
                imu_buf[:] = (roll, pitch, yaw)
                meta_buf[:] = (frame_index, timestamp_ms)
            else:
                with lock:
                    rgb_buf[:] = rgb_mat.get_data()
                    depth_buf[:] = depth_mat.get_data()
                    depth_vision_buf[:] = depth_vision_mat.get_data()
                    imu_buf[:] = (roll, pitch, yaw)
                    meta_buf[:] = (frame_index, timestamp_ms)

            if frame_ready_event is not None:
                frame_ready_event.set()

    except Exception as exc:
        if ready_queue is not None and not ready_sent:
            try:
                ready_queue.put_nowait({"error": str(exc)})
            except Exception:
                pass
        raise

    finally:
        zed.close()

        for shm in (rgb_shm, depth_shm, depth_vision_shm, meta_shm, imu_shm, calib_shm):
            if shm is None:
                continue
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
