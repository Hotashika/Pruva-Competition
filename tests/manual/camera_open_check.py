"""Manual ZED camera open check.

Run explicitly on the machine connected to the camera:

    python tests/manual/camera_open_check.py

Select the competition package when needed:

    COMPETITION_PROJECT=teknofest python tests/manual/camera_open_check.py

This check opens the ZED camera directly through the ZED SDK. If opening fails,
it reports the likely cause together with the raw SDK error or Python traceback.
"""

import importlib
import os
import sys
import traceback
import unittest
from pathlib import Path


COMPETITION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PACKAGE = Path(os.environ.get("COMPETITION_PROJECT", "njord")).name.lower()

if str(COMPETITION_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPETITION_ROOT))


def _load_zed_sdk():
    try:
        import pyzed.sl as sl
    except Exception as exc:
        raise RuntimeError(
            "ZED Python SDK import edilemedi. Muhtemel nedenler: "
            "pyzed wheel yüklü değil, Python sürümü uyumsuz, ZED SDK kurulu değil "
            "veya LD_LIBRARY_PATH ZED kütüphanelerini görmüyor."
        ) from exc
    return sl


def _load_camera_config():
    try:
        return importlib.import_module(f"{PROJECT_PACKAGE}.config.camera_config")
    except Exception as exc:
        raise RuntimeError(
            f"{PROJECT_PACKAGE}.config.camera_config yüklenemedi. Muhtemel nedenler: "
            "yanlış COMPETITION_PROJECT değeri, eksik bağımlılık veya camera_config "
            "içindeki ZED enumlarının mevcut SDK sürümüyle uyumsuz olması."
        ) from exc


def _error_name(status):
    return str(getattr(status, "name", status))


def _camera_open_hint(status_text):
    normalized = status_text.upper()
    if "CAMERA_NOT_DETECTED" in normalized or "NO_GPU_DETECTED" in normalized:
        return "Kamera/USB bağlantısı, güç veya NVIDIA GPU erişimi kontrol edilmeli."
    if "SENSORS_NOT_INITIALIZED" in normalized:
        return "ZED sensörleri başlatılamadı; kamera başka process tarafından kullanılıyor olabilir."
    if "LOW_USB_BANDWIDTH" in normalized or "USB" in normalized:
        return "USB portu/kablosu bant genişliği yetersiz olabilir; USB 3.x port deneyin."
    if "CAMERA_REBOOTING" in normalized:
        return "Kamera yeniden başlıyor; birkaç saniye bekleyip tekrar deneyin."
    if "INVALID_RESOLUTION" in normalized or "INVALID" in normalized:
        return "camera_config çözünürlük/FPS/depth ayarları bu kamera veya SDK ile uyumsuz olabilir."
    if "SUCCESS" in normalized:
        return "Kamera açıldı."
    return "ZED SDK kamerayı açamadı; ham hata aşağıda."


class TestZedCameraOpen(unittest.TestCase):
    def test_zed_camera_opens(self):
        try:
            sl = _load_zed_sdk()
            camera_config = _load_camera_config()

            init = sl.InitParameters()
            init.camera_resolution = camera_config.CAMERA_RESOLUTION
            init.camera_fps = camera_config.CAMERA_FPS
            init.depth_mode = camera_config.DEPTH_MODE
            init.coordinate_units = camera_config.COORDINATE_UNITS
            init.coordinate_system = camera_config.COORDINATE_SYSTEM

            zed = sl.Camera()
            try:
                status = zed.open(init)
                status_text = _error_name(status)

                if status != sl.ERROR_CODE.SUCCESS:
                    self.fail(
                        "\nZED kamera açılamadı."
                        f"\nMuhtemel neden: {_camera_open_hint(status_text)}"
                        f"\nRaw SDK error: {status!r}"
                        f"\nSDK error text: {status_text}"
                        "\nAyarlar:"
                        f"\n  project={PROJECT_PACKAGE}"
                        f"\n  resolution={camera_config.CAMERA_RESOLUTION!r}"
                        f"\n  fps={camera_config.CAMERA_FPS!r}"
                        f"\n  depth_mode={camera_config.DEPTH_MODE!r}"
                        f"\n  coordinate_units={camera_config.COORDINATE_UNITS!r}"
                        f"\n  coordinate_system={camera_config.COORDINATE_SYSTEM!r}"
                    )

                info = zed.get_camera_information()
                serial_number = getattr(info, "serial_number", "unknown")
                camera_model = getattr(info, "camera_model", "unknown")
                print(
                    "\nZED kamera başarıyla açıldı."
                    f"\n  project={PROJECT_PACKAGE}"
                    f"\n  model={camera_model}"
                    f"\n  serial_number={serial_number}"
                )
            finally:
                zed.close()
        except AssertionError:
            raise
        except Exception as exc:
            self.fail(
                "\nZED kamera açılış testi beklenmeyen hata ile durdu."
                f"\nMuhtemel neden: {exc}"
                "\nRaw Python traceback:"
                f"\n{traceback.format_exc()}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
