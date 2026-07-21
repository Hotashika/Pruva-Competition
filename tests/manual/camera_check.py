"""Manual ZED camera integration check; not part of the pytest suite."""

import importlib
import os
import sys
import unittest
from pathlib import Path

import numpy as np

COMPETITION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PACKAGE = Path(os.environ.get("COMPETITION_PROJECT", "njord")).name.lower()

if str(COMPETITION_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPETITION_ROOT))

camera_config = importlib.import_module(f"{PROJECT_PACKAGE}.config.camera_config")
shared_frame_source = importlib.import_module(f"{PROJECT_PACKAGE}.core.shared_frame_source")

CAMERA_HEIGHT = camera_config.CAMERA_HEIGHT
CAMERA_WIDTH = camera_config.CAMERA_WIDTH
close_capture_source = shared_frame_source.close_capture_source
open_or_start_capture_source = shared_frame_source.open_or_start_capture_source


class TestCameraHardware(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Sınıf başlatıldığında frame kaynağına sadece BİR KERE bağlanır."""
        print("\n[TEST] capture_proc frame kaynağına bağlanılıyor...")
        cls.frame_source = None
        cls.capture_process = None
        cls.capture_stop_event = None
        try:
            cls.frame_source, cls.capture_process, cls.capture_stop_event = open_or_start_capture_source()
        except Exception as e:
            print(f"capture_proc frame kaynağına bağlanılamadı: {e}")

    @classmethod
    def tearDownClass(cls):
        """Tüm testler bittikten sonra yerel capture process başlatıldıysa kapatır."""
        close_capture_source(cls.frame_source, cls.capture_process, cls.capture_stop_event)
        print("\n[TEST] capture_proc frame kaynağı güvenli bir şekilde kapatıldı.")

    def test_01_initialization(self):
        """capture_proc shared memory kaynağına başarıyla bağlanıldığını doğrular."""
        self.assertIsNotNone(self.frame_source, "capture_proc frame source None döndürdü.")
        fx, cx = self.frame_source.get_calibration()
        self.assertGreater(fx, 0.0, "Geçersiz fx kalibrasyonu.")
        self.assertGreater(cx, 0.0, "Geçersiz cx kalibrasyonu.")
        print("\n[TEST - 01] capture_proc frame kaynağı başarıyla ilklendirildi.")

    def test_02_frame_grab(self):
        """capture_proc üzerinden görüntü ve depth alınabildiğini doğrular."""
        frame_data = self.frame_source.read(timeout=3.0)
        frame = frame_data["frame_bgr"]
        depth = frame_data["depth"]

        self.assertEqual(frame.shape, (CAMERA_HEIGHT, CAMERA_WIDTH, 3), "RGB frame boyutu hatalı.")
        self.assertEqual(depth.shape, (CAMERA_HEIGHT, CAMERA_WIDTH), "Depth frame boyutu hatalı.")
        self.assertTrue(np.any(frame), "RGB frame boş görünüyor.")
        print("\n[TEST - 02] Frame capture_proc üzerinden başarıyla yakalandı.")


if __name__ == '__main__':
    unittest.main()
