def show_img(self):
    import os
    from PySide2 import QtGui
    list = []
    dir_path = "fireflies/output"
    paths = os.listdir(dir_path)
    if len(paths) == 1:
        dir_path = os.path.join(dir_path, "exp")
    else:
        del paths[0]
        for path in paths:
            path = path.split("p", -1)[1]
            list.append(path)

        max_path = max(list)
        name = "exp" + str(max_path)
        dir_path = os.path.join(dir_path, name)
    imgs = os.listdir(dir_path)
    for img in imgs:
        img_path = os.path.join(dir_path, img)
    img_path = str(img_path)
    pixmap = QtGui.QPixmap(img_path)
    self.ui.label_img.setPixmap(pixmap)
    self.ui.label_img.setScaledContents(True)