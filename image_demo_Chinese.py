#! /usr/bin/env python
# coding=utf-8


import cv2
import numpy as np
import core.utils_Chinese as utils
import tensorflow as tf
from PIL import Image

# 定义基本参数变量
return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file = "./checkpoint/yolov3_coco_v3.pb"   # "./checkpoint/yolov3_helmet.pb"   # 预测文件路径
output_path = './demo_Chinese.jpg'
num_classes = 80                 # 类别数
input_size = 608
graph = tf.Graph()

def predict(image_path):

    original_image = cv2.imread(image_path)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]
    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

    with tf.Session(graph=graph) as sess:
        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.4)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    image = utils.draw_bbox(original_image, bboxes)
    image = Image.fromarray(image)
    image.show()
    image.save(output_path)


if __name__ == "__main__":
    image_path = "./docs/normal_images/yolov.jpg"
    predict(image_path)



