"""Manual GUI video stream check; not part of the pytest suite."""

import sys

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


class VideoThread(QtCore.QThread):
    frame_signal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.url)

        print("VideoCapture acildi mi:", cap.isOpened())

        while self.running:
            ret, frame = cap.read()

            if not ret:
                print("Frame okunamadi")
                self.msleep(100)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape

            image = QtGui.QImage(
                frame.data,
                w,
                h,
                ch * w,
                QtGui.QImage.Format_RGB888
            )

            self.frame_signal.emit(image.copy())

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class TestWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Kamera Test")
        self.resize(800, 600)

        self.label = QtWidgets.QLabel("Goruntu bekleniyor...")
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.thread = VideoThread("http://127.0.0.1:5000/video_feed")
        self.thread.frame_signal.connect(self.update_image)
        self.thread.start()

    def update_image(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        self.label.setPixmap(
            pixmap.scaled(
                self.label.width(),
                self.label.height(),
                QtCore.Qt.KeepAspectRatio
            )
        )

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


app = QtWidgets.QApplication(sys.argv)
window = TestWindow()
window.show()
sys.exit(app.exec_())
