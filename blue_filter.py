import shutil

import cv2 as cv
import numpy as np
import os


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv.filter2D(image, -1, kernel=kernel)

def delete():
    filepath = "fireflies/graph"
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

path = "fireflies/graph"
dir = os.listdir(path)
name = dir[0]
img_path = os.path.join(path,name)
src = cv.imread(img_path)
cv.imwrite("fireflies/img/"+name,src)
delete()
# custom_blur_demo(src)
# cv.waitKey(0)
# cv.destroyAllWindows()