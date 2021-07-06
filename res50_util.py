import numpy as np


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def get_cls_score(outputs):
    # print("outputs",outputs)
    # print("type",type(outputs))
    cls_outputs = softmax(np.array(outputs[0][0]))

    return cls_outputs[1]


def image_process(img):
    # rknn上已经归一化
    img=img/255.0

    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    img[:,:,0] = (img[:,:,0]  - input_mean[0]) / input_std[0]
    img[:,:,1] = (img[:,:,1]  - input_mean[1]) / input_std[1]
    img[:,:,2] = (img[:,:,2] - input_mean[2]) / input_std[2]
    return np.uint8(img * 255.0)
