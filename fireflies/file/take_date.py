def takedate():
    import cv2
    import os

    date = []

    path = "fireflies/img"
    dirs = os.listdir(path)
    for dir in dirs:
        #img_path = os.path.join(path, dir)
        date_num = dir.split(".",-1)[0]
        date.append(date_num)

    if len(date) == len(dirs):
        return date
    else:
        print("有日期未检测出")




