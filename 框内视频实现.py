from PySide2.QtWidgets import QApplication, QMessageBox, QSizePolicy
from PySide2.QtUiTools import QUiLoader
import os
import PySide2
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication, QLabel,QWidget
import time
import cv2
import torch
import numpy
import copy
import matplotlib.mlab as mlab
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from PySide2.QtGui import QPixmap
import threading
from PySide2.QtGui import *


class Stats():

    def __init__(self):
        # 从文件中加载UI定义

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load('ui/main.ui')
        self.timer_camera = QTimer()

        #self.ui.label_img.setFixedSize(self.ui.label_img.width(), self.ui.label_img.height())
        #self.ui.label_graph.setFixedSize(self.ui.label_graph.width(), self.ui.label_graph.height())
        #self.ui.button.clicked.connect(self.detection)
        #self.ui.button_show.clicked.connect(self.show_img)

    def Open(self):
        self.cap = cv2.VideoCapture(0)
        self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
            # 创建视频显示线程
        th = threading.Thread(target=self.Display)
        th.start()

    def Display(self):
        num=0
        while self.cap.isOpened():
            success, frame = self.cap.read()
                    # RGB转BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.label_graph.setPixmap(QPixmap.fromImage(img))
                    # 按比例填充
            self.ui.label_graph.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            self.ui.label_graph.setScaledContents(True)
            k = cv2.waitKey(1) & 0xFF  # 按键判断
            if k == ord('s'):  # 保存
                cv2.imwrite("fireflies/aaa/" + str(num) + "." + ".jpg", frame)
                print("success to save" + str(num) + ".jpg")
                print("-------------------")
                num += 1
            elif k == ord(' '):  # 退出
                break


if __name__ == '__main__':
    app = QApplication([])
    video = Stats()
    video.ui.show()
    #video.Open()
    app.exec_()
