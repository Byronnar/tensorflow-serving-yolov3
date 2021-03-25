#! /usr/bin/env python
# coding=utf-8
import cv2
import os,glob
import numpy as np
import core.utils as utils
import tensorflow as tf
import time

s1 = time.time()
def batch_infer(img_dir="./docs/normal_images", output_path='./output'):
    # 指定第一个文件夹的位置
    imageDir = os.path.abspath(img_dir)

    # 通过glob.glob来获取第一个文件夹下，所有'.jpg'文件
    imageList = glob.glob(os.path.join(imageDir, '*.jpg'))
    # print(imageList)
    imgs_num = len(imageList)

    graph = tf.Graph()
    pb_file = "./checkpoint/yolov3_coco_v3.pb"
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)  # 读取刚刚变量

    with tf.Session(graph=graph) as sess:       # 要有这种思想，一个会话处理全部图片。

        for item in imageList:
            image_path      = item
            # print('item',item)
            end = "/"
            name = item[item.rfind(end):] # 获取图片文件名
            # print(name)
            num_classes     = 80
            input_size      = 608
            out =output_path + name

            original_image = cv2.imread(image_path)
            # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image_size = original_image.shape[:2]
            image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size]) #图片处理
            image_data = image_data[np.newaxis, ...]

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0) #整合预测框


            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.45) # 这一步是将所有可能的预测信息提取出来，主要是三类：坐标值，可能性，类别
            # 如果出现错认的情况，请把 0.76调高到8试试，input_size 改为416

            # print('bboxes:',bboxes)
            # bboxes: [[301.13088989 118.44700623 346.95623779 172.39486694   0.97461057   0]...]

            bboxes = utils.nms(bboxes, 0.45, method='nms') # 这一步是 将刚刚提取出来的信息进行筛选，返回最好的预测值，同样是三类。
            # print('bboxes:',bboxes)
            # bboxes: [array([105.31238556,  54.51167679, 282.53552246, 147.27146912, 0.99279714,   0.        ])]

            image = utils.draw_bbox(original_image, bboxes) # 这一步是把结果画到新图上面

            cv2.imwrite(out, image) # 保存检测结果

    return imgs_num

if __name__ == "__main__":

    img_dir = "./docs/normal_images"
    imgs_num = batch_infer(img_dir)
    s2 = time.time()
    print('图片数量为: ', imgs_num, '预测总用时： ', s2 - s1)
