# 改为多线程
from __future__ import division
import os
import time
import threading
import queue
import cv2
import numpy as np

from rknn.api import RKNN  ##package for rk3399prod
from yolo_util import yolov3_post_process, letterbox_image
from res50_util import get_cls_score, image_process

# yolo config
GRID0 = 13
GRID1 = 26
GRID2 = 52
LISTSIZE = 6
SPAN = 3
NUM_CLS = 1
MAX_BOXES = 500
OBJ_THRESH = 0.5
NMS_THRESH = 0.4

#
# confidence = 0.6
# nms_thesh = 0.4
FALL_TH = 0.2  # 顶格相当于全局变量


def get_bboxes(image, boxes, scores, classes):
    new_boxes = []
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x + w, y + h))
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

        # print('class: {0}, score: {1:.2f}'.format(CLASSES[cl], score))
        # print('box coordinate x,y,w,h: {0}'.format(box))
        x1, y1, x2, y2 = top, left, right, bottom
        new_boxes.append([x1, y1, x2, y2])
    return new_boxes


def filter_personbbox(bbox):
    '''
    :param bbox:
    :return: True of Flase
    '''
    x, y = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
    eage = x < 300 or y < 300 or x > 1920 - 300 or y > 1080 - 300
    h, w, radio = bbox[3] - bbox[1], bbox[2] - bbox[0], (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])
    # return h < 50 or w < 50 or radio > 3 #or eage
    return h < 60 or w < 150 or radio > 3 or eage


def run_detect(input_img, yolo_rknn, res50_rknn):
    # input_img=cv2.imread('./person_416x416.jpg')
    orig_img = input_img
    num_classes = 1
    bbox_attrs = 5 + num_classes

    yolo_rknn = yolo_rknn

    res50_rknn = res50_rknn

    # input_transforms = build_transforms(cfg, is_train=False)  #分类图像减去均值除以方差，所以要自己处理一下

    # model.net_info["height"] = 416
    # inp_dim = int(model.net_info["height"])
    # assert inp_dim % 32 == 0
    # assert inp_dim > 32

    frames = 0
    start = time.time()
    ## ToDo ADD here by wzw
    input_img = (letterbox_image(input_img, (416, 416)))
    # input_img = input_img / 255.0
    input_img = np.uint8(input_img)
    # print(input_img[200][200])

    out_boxes, out_boxes2, out_boxes3 = yolo_rknn.inference(inputs=[input_img])

    out_boxes = out_boxes.reshape(SPAN, LISTSIZE, GRID0, GRID0)
    out_boxes2 = out_boxes2.reshape(SPAN, LISTSIZE, GRID1, GRID1)
    out_boxes3 = out_boxes3.reshape(SPAN, LISTSIZE, GRID2, GRID2)
    input_data = []
    input_data.append(np.transpose(out_boxes, (2, 3, 0, 1)))
    input_data.append(np.transpose(out_boxes2, (2, 3, 0, 1)))
    input_data.append(np.transpose(out_boxes3, (2, 3, 0, 1)))

    boxes, classes, scores = yolov3_post_process(input_data)
    # print(boxes)

    # cv2.imwrite(str(scores)+".jpg",orig_img)

    is_fall = False
    cur_frame_valis_bbox = []
    fall_score = 0
    # for i, box in enumerate(bboxes):
    box = []
    if boxes is not None:
        new_boxes = get_bboxes(orig_img, boxes, scores, classes)
        for i, box in enumerate(new_boxes):
            if filter_personbbox(box):
                continue
            # print("ccccccc")
            person_bbox = orig_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            person_bbox = cv2.resize(person_bbox, (64, 64))

            ## ToDo ADD here by wzw
            # person_bbox = image_process(person_bbox)  ###  transforms.Normalize(cfg.input_mean, cfg.input_std)
            # fall_score = inference(person_bbox, model_fall, device, input_transforms)[1]
            print("np mean person ", np.mean(person_bbox))

            outputs = res50_rknn.inference(inputs=[person_bbox])
            print("outputs", outputs)

            fall_score = get_cls_score(outputs)  # top1 score 这里需要去函数打印看一下，改对   ## ToDo
            print("fall_score", fall_score)
            if fall_score > FALL_TH:
                is_fall = True
                cur_frame_valis_bbox.append(box)

    return is_fall, orig_img, box, fall_score


def load_yolo_model():
    rknn = RKNN()
    print('-->loading yolo  model')
    # rknn.load_rknn('./yolov3_tiny.rknn')
    rknn.load_rknn('./yolov3_416x416.rknn')
    print('loading yolo model done')

    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn


def load_resnet50_model():
    rknn = RKNN()
    print('-->loading resnet50  model')
    rknn.load_rknn('./resnet_50.rknn')
    print('loading resnet50 model done')

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    return rknn


def run_cam_multi():
    # 加载模型只需要加载一次，所以放到主函数

    yolo_rknn = load_yolo_model()
    res50_rknn = load_resnet50_model()

    # read  image
    global cap
    cap = cv2.VideoCapture("./our_test_video.mp4")
    assert cap.isOpened(), 'Cannot capture source'
    frame1 = 0

    num_fall_frames = 0
    num_no_fall_frames = 0

    while cap.isOpened():
        ret, input_img = cap.read()
        if ret:
            print("-----------------------------")
            print(frame1)

            # img, orig_im, dim = prep_image(input_img, 416)
            if frame1 % 2 == 0:

                is_fall, draw_img, box, fall_score = run_detect(input_img, yolo_rknn, res50_rknn)

                if is_fall:
                    num_fall_frames += 1
                    # cv2.putText(orig_im, "Fall", (100, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                    #             [0, 0, 255], 3)
                    if len(box) != 0:
                        cv2.rectangle(draw_img, tuple(box[0:2]), tuple(box[2:4]), (0, 0, 255), 2)
                    cv2.putText(draw_img, "FALL", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                                [0, 0, 255], 2)
                    print("FALL")
                else:
                    if len(box) != 0:
                        cv2.rectangle(draw_img, tuple(box[0:2]), tuple(box[2:4]), (0, 255, 0), 1)
                    # cv2.putText(draw_img, "STAND", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                    #             [0, 255, 0], 2)
                    num_no_fall_frames += 1

                if num_no_fall_frames >= 4:
                    # no_fall
                    num_fall_frames = 0
                    num_no_fall_frames = 0

                cv2.imshow("fall_detect", draw_img)
                cv2.waitKey(10)
                # 获取窗口大小
                size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                if frame1 == 0:
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    writer = cv2.VideoWriter('fall_detect_demo.avi', fourcc, 6, size)

                writer.write(draw_img)
            frame1 = frame1 + 1



if __name__ == '__main__':
    run_cam_multi()
