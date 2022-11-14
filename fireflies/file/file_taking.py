import time
import cv2
from PySide2.QtWidgets import QFileDialog
from fireflies.file.show_img_taking import show_img_taking

def file_taking(self):
    filePath, _ = QFileDialog.getOpenFileName(self.ui, "选择你要上传的图片", "./", "*.*")
    print(filePath)
    img = cv2.imread(filePath)
    cv2.imwrite("fireflies/img/" + time.strftime("%Y-%m-%d", time.localtime()) + "." + str(self.num) + ".jpg", img)
    show_img_taking(self)