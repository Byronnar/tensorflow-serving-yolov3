#! /usr/bin/env python
# coding=utf-8


import cv2
import time
import numpy as np
import core.utils_Chinese as utils
import tensorflow as tf
from PIL import Image

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./checkpoint/yolov3_coco_v3.pb"
num_classes     = 80
input_size      = 416
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

def infer_video(video_path, output_path):
    with tf.Session(graph=graph) as sess:
        writeVideo_flag = True
        if writeVideo_flag:
            vid = cv2.VideoCapture(video_path)
            if not vid.isOpened():
                raise IOError("Couldn't open webcam or video")
            video_FourCC = cv2.VideoWriter_fourcc(*'MP4V')
            video_fps       = vid.get(cv2.CAP_PROP_FPS)
            video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            isOutput = True if output_path != "" else False
            if isOutput:
                #print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
                out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
                list_file = open('detection.txt', 'w')
                frame_index = -1
        while True:
            return_value, frame = vid.read()
            if return_value != True:
                break
            if return_value:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                # print('image:',image)
            else:
                raise ValueError("No image!")
            frame_size = frame.shape[:2]
            image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]

            prev_time = time.time()

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={ return_tensors[0]: image_data})
            pred_time = time.time()

            # print('time:',pred_time-prev_time)

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.45)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            image = utils.draw_bbox(frame, bboxes)

            curr_time = time.time()
            exec_time = curr_time - prev_time
            result = np.asarray(image)
            info = "time:" + str(round(1000 * exec_time, 2)) + " ms, FPS: " + str(round((1000 / (1000 * exec_time)), 1))
            cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=2) # 把字加上去
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if writeVideo_flag:
                # save a frame
                out.write(result)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

if __name__ == "__main__":

    video_path = "./docs/video/car.mp4"
    # video_path      = 0   # 开启摄像头
    output_path = './output/demo_Chinese.mp4'

    infer_video( video_path, output_path )
