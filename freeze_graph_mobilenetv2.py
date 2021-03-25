#! /usr/bin/env python
# coding=utf-8
# 固化 mobilenetv2 模型

import tensorflow as tf
from core.yolov3_mobilenetv2 import YOLOV3

def model_freeze(pb_file, ckpt_file):
    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')

    model = YOLOV3(input_data, trainable=tf.cast(False,dtype=tf.bool))                 # 加载yolov3网络

    sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

    converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                input_graph_def  = sess.graph.as_graph_def(),
                                output_node_names = output_node_names)

    with tf.gfile.GFile(pb_file, "wb") as f:
        f.write(converted_graph_def.SerializeToString())


if __name__ == "__main__":
    
    pb_file = "./checkpoint/yolov3_mobilenetv2.pb"
    ckpt_file = "./checkpoint/yolov3_mobilenetv2.ckpt"

    output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2",
                         "pred_multi_scale/concat"]

    model_freeze(pb_file, ckpt_file)
    print('Mobilenet 模型固化已完成.')




