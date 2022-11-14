import time

import cv2
import pyzbar
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtWidgets import QSizePolicy


def display(self):
    self.cap = cv2.VideoCapture(0)
    while self.cap.isOpened():
        success, frame = self.cap.read()
        # RGB转BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        test = pyzbar.decode(frame)
        for tests in test:
            self.ui.edit_taking.setText("不可拍照")
            # 获取到的条形码数据转换成字符串
            testdate = tests.data.decode('utf-8')
            if testdate is not None:
                self.ui.edit_taking.setText("可以拍照")
                time.sleep(0.1)
                self.ui.edit_taking.setText(" ")#检测是否可以检测到二维码

            # else:
            #     self.ui.edit_taking.setText("不可拍照")
        img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.ui.label_video.setPixmap(QPixmap.fromImage(img))
        # 按比例填充
        self.ui.label_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.ui.label_video.setScaledContents(True)

        self.cap.release()