#! /usr/bin/env python
# coding=utf-8


import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from config import cfg


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def image_preporcess(image, target_size, gt_boxes=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)  # 类别数
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))  # 转换格式
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:  # 显示标签
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)  # 转化为数组
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])  # 选出最大值
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def read_pb_return_tensors(graph, pb_file, return_elements):
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):  # iou_threshold是指两个框重合度 阈值
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class) # 上一步生成的

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    # all the classes from bboxes
    classes_in_img = list(set(bboxes[:, 5]))  # 第五列代表图片里面的类别
    best_bboxes = []

    for cls in classes_in_img:  # 挑出目标图片里面存在的框
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])  # 返回最大值，即可能性最高的
            best_bbox = cls_bboxes[max_ind]  # 最佳的框
            best_bboxes.append(best_bbox)  # 框增加
            # find the best bbox with the max score for one class and remove it from
            # 只保留最佳预测结果
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])  # 选出最佳的框
            weight = np.ones((len(iou),), dtype=np.float32)  # ones函数可以创建任意维度和元素个数的数组，其元素值均为1；

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                # remove the bboxes that overlap with the best bbox beyond the threshold
                # this removes the duplicate bboxes generated for the same object
                # there may be bboxes with high score for the same type of object in the image
                iou_mask = iou > iou_threshold  # 选择最好的阈值
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes  # 返回最好的框预测信息


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)  # pred_bbox就是刚刚传进来的output

    pred_xywh = pred_bbox[:, 0:4]  # 进行切片 行：全部，列：前四列  中心点坐标
    pred_conf = pred_bbox[:, 4]  # 代表第四列 4 （代表是某类的可能性）
    pred_prob = pred_bbox[:, 5:]  # 第五列之后 5-85 （代表 是哪一类）

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax) 计算出这四个值
    pred_coor = np.concatenate(
        [pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,  # 计算x y坐标减去 w h乘以0.5的差（xmin.ymin)以及 和(xmax,ymax)
         pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org) 获取原图四个处理后的坐标值
    org_h, org_w = org_img_shape  # 原图尺寸
    resize_ratio = min(input_size / org_w, input_size / org_h)  # 输入尺寸除以原图尺寸并求最小值

    # print('resize_ratio', resize_ratio)

    dw = (input_size - resize_ratio * org_w) / 2  # 计算差值
    dh = (input_size - resize_ratio * org_h) / 2
    # print(org_w)  # 500
    # print(org_h)  # 375
    # print(dw)  # 0.0
    # print(dh)  # 76.0 保留一位小数

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio  # 0::2代表 start : end : step
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
    # 例如：
    # >>> a = [1,2,3,4,5,6,7,8,9]
    # >>> a[::3]
    # [1, 4, 7]

    # print('pred_coor[:, 0::2]',pred_coor[:, 0::2])
    # print('pred_coor[:, 0::2]', pred_coor[:, 1::2])

    # # (3) clip some boxes those are out of range 舍弃超过边界的框，即出现 xmin>xmax等类似情况舍去
    pred_coor = np.concatenate(
        [np.maximum(pred_coor[:, :2], [0, 0]),  # Join a sequence of arrays along an existing axis.即整合最大小值
         np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])],
        axis=-1)  # axis=-1，其实也就等于axis=2，设axis=i，则沿着第i个下标变化的方向进行操作

    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]),
                                 (pred_coor[:, 1] > pred_coor[:, 3]))  # 找出xmin>xmax等类似情况
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes 舍弃无效框
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))  # sqrt 求平方根函数
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale),
                                (bboxes_scale < valid_scale[1]))  # logical_and逻辑与，判断是否正确

    # # (5) discard some boxes with low scores 舍弃可能性低的框 pred_conf置信度 低于score_threshold这个舍弃
    classes = np.argmax(pred_prob, axis=-1)  # argmax取出最大值，即某类可能性最大的值
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold

    mask = np.logical_and(scale_mask, score_mask)  # logical_and逻辑与，判断是否正确

    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]  # 三个主要信息 （框，可能性，类别）

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)  # 整合在一起（框，可能性，类别）


