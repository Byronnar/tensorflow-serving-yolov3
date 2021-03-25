#! /usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

def read_pb_return_tensors_mobilenet(graph, pb_file, ori_return_elements): # mobilenet 的tensor处理
    with graph.as_default():
        with tf.gfile.FastGFile( pb_file, 'rb' ) as f:
            frozen_graph_def = tf.GraphDef()
            frozen_graph_def.ParseFromString( f.read() )
        # fix nodes
        for node in frozen_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range( len( node.input ) ):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            elif node.op == 'Assign':
                node.op = 'Identity'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
                if 'validate_shape' in node.attr:
                    del node.attr['validate_shape']
                if len( node.input ) == 2:
                    node.input[0] = node.input[1]
                    del node.input[1]

    with graph.as_default():
        return_elements = tf.import_graph_def( frozen_graph_def, return_elements=ori_return_elements )

    return return_elements


def predict(image_path):

    original_image = cv2.imread(image_path) # 读取图片
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]

    return_tensors = read_pb_return_tensors_mobilenet(graph, pb_file, ori_return_elements)

    with tf.Session(graph=graph) as sess:
        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})


    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.35)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    image = utils.draw_bbox(original_image, bboxes)

    image = Image.fromarray(image)
    image.show()
    image.save(output_path)


if __name__ == "__main__":

    # 定义基本参数变量
    ori_return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]

    pb_file = "./checkpoint/yolov3_stu.pb"  # "./checkpoint/yolov3_helmet.pb"   # 预测文件路径
    output_path = './demo.jpg'
    num_classes = 2  # 类别数
    input_size = 416
    graph = tf.Graph()

    # image_path = "./docs/normal_images/000000000080.jpg" # 这里输入预测图片路径
    image_path = "./VOC2007/JPEGImages/0.jpg"  # 这里输入预测图片路径

    predict(image_path)




