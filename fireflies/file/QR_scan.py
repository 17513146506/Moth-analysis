import shutil


def QR_scan():
    import cv2
    from pyzbar import pyzbar
    import csv
    import os

    label = []

    # 然后我们设置一个变量，来存放我们扫到的码的信息，我们每次扫描一遍都会要检测扫描到的码是不是之前扫描到的，
    # 如果没有就存放到这里。接着我们调用opencv的方法来实例化一个摄像头，
    # 最后我们设置一些我们存放码信息的表格的路径。
    #found = set()
    # 打开摄像头，0代表本地摄像头
    #capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # 不停的用摄像头来采集条码，



    path = "fireflies/img"

    def delete():
        filepath = "fireflies/img"
        del_list = os.listdir(filepath)
        for f in del_list:
            file_path = os.path.join(filepath, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    dirs = os.listdir(path)


    for dir in dirs:
        img = cv2.imread(os.path.join(path,dir))
        test = pyzbar.decode(img)

        for tests in test:
            # 获取到的条形码数据转换成字符串
            testdate = tests.data.decode('utf-8')
            if testdate:
                label.append(testdate)

        print("a")

    if len(label) == len(dirs):
        return label
    else:
        print("有二维码未检测出")
        delete()

