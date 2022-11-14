# -*- coding: UTF-8 -*-
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


graph = [[0],
              [0],
              [0]]
graph = numpy.array(graph)

labels = [['fireflies'],
          ["ants"],
          ["butterflies"]]
labels = numpy.array(labels)


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def to_graph(c,data):
    graph[c,0] += data

def draw_graph(graph,labels):
    X = [labels[0,0],labels[1,0],labels[2,0]]
    Y = [graph[0,0],graph[1,0],graph[2,0]]
    fig = plt.figure()
    plt.bar(X, Y, 0.4, color="green")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("bar chart")

    plt.show()
    plt.savefig("fireflies/graph/barChart.jpg")

def show_results(img, xywh, conf, class_num):
    h, w, c = img.shape
    labels = ['fire']
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    tf = max(tl - 1, 1)  # font thickness
    # label = str(labels[int(class_num)]) + ': ' + str(conf)[:5]
    label = str(labels[int(class_num)])
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [0, 0, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect_one(model, image_path, device):
    # Load model
    img_size = 320#要转化为320*320的图像
    conf_thres = 0.4
    iou_thres = 0.2

    orgimg = image_path
    img0 = copy.deepcopy(orgimg)#复制图像，防止对原图造成影响
    assert orgimg is not None, 'Image Not Found ' + image_path#检测是否存在这个图像
    h0, w0 = orgimg.shape[:2]  # orig hw  #获得shape中的两个值
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation    #判断原图的大小与要转换的图的大小关系
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR #如果r小则要使用resize中的扩大参数，若果r大则设置为调小参数
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)#更改大小（按照比例缩放，并且使最大的边等于320）

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size #看设置的大小是否为32可除，若不是则改为32可整除

    img = letterbox(img0, new_shape=imgsz)[0]#使用letterbox用单一的像素点补全不够320的边
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416  #第一个中括号，代表BGR转化为RGB，

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)#转化为tensor
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0 #把0 - 255的数值转化为 0.0 - 1.0的数值
    if img.ndimension() == 3:#如果维度等于3
        img = img.unsqueeze(0)#那么在0的位置（第一个位置）增加一个维度

    # Inference
    pred = model(img)[0]

    # Apply NMS


    pred = non_max_suppression(pred, conf_thres, iou_thres)#非极大值抑制，排除同一个物体上冗余的框框
    print(len(pred))
    # pred: 网络的输出结果
    # conf_thres: 置信度阈值
    # ou_thres: iou阈值
    # classes: 是否只保留特定的类别
    # agnostic_nms: 进行nms是否也去除不同类别之间的框
    # max - det: 保留的最大检测框数量
    # ---NMS, 预测框格式: xywh(中心点 + 长宽) -->xyxy(左上角右下角)
    # pred是一个列表list[torch.tensor], 长度为batch_size
    # 每一个torch.tensor的shape为(num_boxes, 6), 内容为box + conf + cls
    print('pred: ', pred)
    print('img.shape: ', img.shape)
    print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image  #组合为一个索引序列，i为索引表，det为对应数值的表，
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh  #。to(device)指把这些数据复制一份放到gpu上
        if len(det):
            # 得到的图像还原回原来的图像上面，即 将预测信息映射到原图
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()# 每一类都检测
                c = int(c)
                n = int(n)
                to_graph(c,n)
                print(graph)
                # print("_____")
                # print(int(c))
                # print(n)


            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(torch.tensor(det[j, :4]).view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                class_num = det[j, 4].cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, class_num)



    # Stream results
    print(f'Done. ({time.time() - t0:.3f}s)')
    return orgimg


# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     weights = 'fireflies/weight/best.pt'
#     model = load_model(weights, device)
#
#     img = cv2.imread("fireflies/img/img.png")
#     img = detect_one(model, img, device)
#     draw_graph(graph,labels)
#     cv2.imshow("img", img)
#     cv2.waitKey(0)

    # using camera
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     _, frame = cap.read()
    #     frame = detect_one(model, frame, device)
    #     cv2.imshow("img", frame)
    #     cv2.waitKey(1)
    # print('over')

