import shutil
from PySide2 import QtGui
from PySide2.QtUiTools import QUiLoader
import os
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication, QLabel,QWidget
import cv2
import matplotlib
from fireflies.file.display import display
from fireflies.file.img_taking import taking
matplotlib.use('TkAgg')
import threading
from detection import detection
from fireflies.file.create_txt import create_txt
from fireflies.file.create_dir import create_dir
from fireflies.file.QR_scan import QR_scan
from fireflies.file.take_date import takedate
from fireflies.file.file_taking import file_taking
from fireflies.file.show_img_detect import show_img


import sys
sys.setrecursionlimit(10000)

class Stats():

    def __init__(self):

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load('ui/main_2.ui')
        self.timer_camera = QTimer()
        self.cap = cv2.VideoCapture(0)
        #self.ui.button_cam.clicked.connect(self.open)
        self.ui.button_takephoto.clicked.connect(self.taking)
        self.ui.button_exit.clicked.connect(self.exit)
        self.ui.button_detect.clicked.connect(self.detection)
        self.ui.button_file.clicked.connect(self.file)

        self.pixmap = QtGui.QPixmap("ui/background/20221106203948.png")
        self.video = threading.Thread(target=self.display)
        self.taking = threading.Thread(target=self.taking)

        self.ui.label_background.setPixmap(self.pixmap)
        self.ui.label_background.setScaledContents(True)


        self.num = 1

    def display(self):
        display(self)#在框内显示视频


    def taking(self):
        taking(self)#拍照并将拍到的图片显示到label内


    def exit(self):
        exit(0)#退出



    def delete(self):
        #删除img文件夹内的图片
        filepath = "fireflies/img"
        del_list = os.listdir(filepath)
        for f in del_list:
            file_path = os.path.join(filepath, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def detection(self):
        #blue_filter()
        num = detection()
        label = QR_scan()
        date = takedate()
        print(num)
        print(label)
        print(date)
        create_txt(date,label,num)
        create_dir(date)
        show_img(self)#显示检测后的图像
        self.delete()
        self.ui.edit_qr.setText(label[0])#显示编号
        self.ui.edit_num.setText(str(num[0]))#显示数量

    def file(self):
        file_taking(self)





if __name__ == '__main__':
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    #stats.video.start()
    app.exec_()