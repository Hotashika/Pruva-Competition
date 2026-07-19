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
from njord.core.capture_dataset import CaptureDatasetSession


def _enum_text(value):
    return str(getattr(value, "name", value))


def _camera_intrinsics_payload(camera_parameters):
    payload = {
        "fx": float(camera_parameters.fx),
        "fy": float(camera_parameters.fy),
        "cx": float(camera_parameters.cx),
        "cy": float(camera_parameters.cy),
    }
    distortion = getattr(camera_parameters, "disto", None)
    if distortion is not None:
        payload["distortion"] = (
            np.asarray(distortion, dtype=float).reshape(-1).tolist()
        )
    return payload


def _calibration_payload(camera_information):
    parameters = camera_information.camera_configuration.calibration_parameters
    payload = {
        "camera_model": "ZED",
        "resolution": {
            "width": int(RGB_SHAPE[1]),
            "height": int(RGB_SHAPE[0]),
        },
        "camera_fps": int(CAMERA_FPS),
        "coordinate_units": _enum_text(COORDINATE_UNITS),
        "coordinate_system": _enum_text(COORDINATE_SYSTEM),
        "left": _camera_intrinsics_payload(parameters.left_cam),
        "right": _camera_intrinsics_payload(parameters.right_cam),
    }

    try:
        translation = parameters.stereo_transform.get_translation()
        if hasattr(translation, "get"):
            translation = translation.get()
        translation = np.asarray(translation, dtype=float).reshape(-1)
        if translation.size >= 3:
            payload["stereo_translation"] = translation[:3].tolist()
            payload["baseline_m"] = float(np.linalg.norm(translation[:3]))
    except (AttributeError, TypeError, ValueError):
        # Some ZED SDK versions do not expose stereo translation through the
        # Python API. Left/right intrinsics are still sufficient to identify
        # the calibration used for this run.
        pass
    return payload


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
        dataset_output_root=None,
        dataset_task=None,
        dataset_record_fps=5.0,
        dataset_record_event=None,
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
    dataset_session = None
    dataset_start_error = None
    dataset_start_blocked = False
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
        calibration_parameters = cam_info.camera_configuration.calibration_parameters
        left_calibration = calibration_parameters.left_cam
        fx = float(left_calibration.fx)
        fy = float(left_calibration.fy)
        cx = float(left_calibration.cx)
        cy = float(left_calibration.cy)

        runtime = sl.RuntimeParameters()

        rgb_mat = sl.Mat()
        right_mat = sl.Mat()
        depth_mat = sl.Mat()
        depth_vision_mat = sl.Mat()
        sensors_data = sl.SensorsData()
        dataset_calibration = _calibration_payload(cam_info)

        if dataset_output_root is not None and dataset_record_event is None:
            try:
                dataset_session = CaptureDatasetSession(
                    dataset_output_root,
                    task_name=dataset_task,
                    calibration=dataset_calibration,
                    record_fps=dataset_record_fps,
                    record_right=True,
                )
                print(
                    f"[DATASET] {dataset_session.task_name} recording enabled: "
                    f"{dataset_session.run_dir}"
                )
            except Exception as exc:
                dataset_start_error = str(exc)
                dataset_start_blocked = True
                print(f"[DATASET] Recording could not be started: {exc}")
        elif dataset_output_root is not None:
            print(f"[DATASET] Waiting for active task: {dataset_task}")

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
            ready_payload = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
            if dataset_session is not None:
                ready_payload["dataset_run_dir"] = str(dataset_session.run_dir)
            elif dataset_output_root is not None and dataset_record_event is not None:
                ready_payload["dataset_waiting_for_task"] = str(dataset_task)
            if dataset_start_error is not None:
                ready_payload["dataset_error"] = dataset_start_error
            ready_queue.put(ready_payload)
            ready_sent = True

        # ------------------------------------------------------------------
        # Capture Loop
        # ------------------------------------------------------------------
        while stop_event is None or not stop_event.is_set():
            if dataset_output_root is not None:
                recording_requested = (
                    dataset_record_event is None or dataset_record_event.is_set()
                )
                if not recording_requested:
                    dataset_start_blocked = False
                    if dataset_session is not None:
                        closing_session = dataset_session
                        dataset_session = None
                        try:
                            closing_session.close()
                            print(
                                "[DATASET] Task recording finalized: "
                                f"{closing_session.run_dir}"
                            )
                        except Exception as exc:
                            print(f"[DATASET] Recording finalization failed: {exc}")
                elif dataset_session is None and not dataset_start_blocked:
                    try:
                        dataset_session = CaptureDatasetSession(
                            dataset_output_root,
                            task_name=dataset_task,
                            calibration=dataset_calibration,
                            record_fps=dataset_record_fps,
                            record_right=True,
                        )
                        print(
                            f"[DATASET] {dataset_session.task_name} recording enabled: "
                            f"{dataset_session.run_dir}"
                        )
                    except Exception as exc:
                        dataset_start_blocked = True
                        print(f"[DATASET] Recording could not be started: {exc}")

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
            left_image = rgb_mat.get_data()
            if lock is None:
                rgb_buf[:] = left_image
                depth_buf[:] = depth_mat.get_data()
                depth_vision_buf[:] = depth_vision_mat.get_data()
                imu_buf[:] = (roll, pitch, yaw)
                meta_buf[:] = (frame_index, timestamp_ms)
            else:
                with lock:
                    rgb_buf[:] = left_image
                    depth_buf[:] = depth_mat.get_data()
                    depth_vision_buf[:] = depth_vision_mat.get_data()
                    imu_buf[:] = (roll, pitch, yaw)
                    meta_buf[:] = (frame_index, timestamp_ms)

            if dataset_session is not None:
                try:
                    if dataset_session.frame_is_due(timestamp_ms):
                        zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
                        dataset_session.record_frame(
                            frame_id=frame_index,
                            camera_timestamp_ms=timestamp_ms,
                            left_image=left_image,
                            right_image=right_mat.get_data(),
                            roll=roll,
                            pitch=pitch,
                            yaw=yaw,
                        )
                except Exception as exc:
                    print(f"[DATASET] Recording disabled after write failure: {exc}")
                    try:
                        dataset_session.close(timeout=2.0)
                    except Exception as close_exc:
                        print(f"[DATASET] Recorder close failed: {close_exc}")
                    dataset_session = None
                    dataset_start_blocked = True

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
        if dataset_session is not None:
            try:
                dataset_session.close()
                print(f"[DATASET] Recording finalized: {dataset_session.run_dir}")
            except Exception as exc:
                print(f"[DATASET] Recording finalization failed: {exc}")

        zed.close()

        for shm in (rgb_shm, depth_shm, depth_vision_shm, meta_shm, imu_shm, calib_shm):
            if shm is None:
                continue
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
