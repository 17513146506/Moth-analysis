import time
import cv2
from fireflies.file.show_img_taking import show_img_taking


def taking(self):
    success, frame = self.cap.read()
    # RGB转BGR
    k = cv2.waitKey(1) & 0xFF  # 按键判断
    cv2.imwrite("fireflies/img/" + time.strftime("%Y-%m-%d", time.localtime()) + "." + str(self.num) + ".jpg", frame)
    print("success to save" + time.strftime("%Y-%m-%d", time.localtime()) + str(self.num) + ".jpg")
    print("-------------------")
    self.num += 1
    show_img_taking(self)