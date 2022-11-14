from PySide2 import QtGui


def show_img_taking(self):
    from PySide2 import QtGui
    import os
    dir_path = "fireflies/img"
    paths = os.listdir(dir_path)
    img_path = os.path.join(dir_path, paths[0])
    pixmap = QtGui.QPixmap(img_path)
    self.ui.label_img.setPixmap(pixmap)
    self.ui.label_img.setScaledContents(True)