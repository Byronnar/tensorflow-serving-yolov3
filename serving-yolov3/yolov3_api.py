# Api调用demo，还是用到了内置函数
import cv2
from matplotlib import pyplot as plt
import utils as utils
import numpy as np
import requests
import json
from PIL import Image

def object_detect(input_path="./road.jpg", output_path='./demo.jpg'):
    img_size = 608
    num_channels = 3
    # image_path = "./docs/images/sample_computer.jpg"
    image_path = input_path # 调用图片,示例："./docs/images/sample_computer.jpg"
    original_image = cv2.imread(image_path)

    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = utils.image_preporcess(np.copy(original_image), [img_size, img_size])# 图片处理成608*608*3

    # print(image_data.shape)

    plt.imshow(image_data)
    plt.show()


    yolov3_api = "http://localhost:8501/v1/models/yolov3:predict"   # 刚刚产生的接口
    image_data_yolo_list = image_data[np.newaxis, :].tolist() # 转化为多维数组矩阵


    headers = {"Content-type": "application/json"}
    r = requests.post(yolov3_api, headers=headers,
                      data=json.dumps({"signature_name": "predict",
                                       "instances": image_data_yolo_list})).json() 	#post请求

    # print('r',r) # 19, 19, 85 = 30685
    # {'error': 'Input to reshape is a tensor with 18411 values, but the requested shape requires a multiple of 30685\n\t [[{{node pred_multi_scale/Reshape_2}}]]'}
    # 18411 的因子 [3, 17, 19, 51, 57, 323, 361, 969, 1083, 6137]

    output = np.array(r['predictions'])
    # print(output.shape)
    #   (63, 19, 19, 85)  reduction factor 注：衰减系数以及步长：32  608/32=19      85 = 80类+1可能性+4个坐标
    #   416 x 416 则为 13*13

    output = np.reshape(output, (-1, 85)) # 这一步处理成 22743*85的维度（63*19*19 =22743， 85 = 80类+1可能性+4个坐标,根据自己数据集改）
    # print(output.shape)

    original_image_size = original_image.shape[:2]
    bboxes = utils.postprocess_boxes(output, original_image_size, img_size, 0.3)  # 这一步是将所有可能的预测信息提取出来，主要是三类：类别，可能性，坐标值。
    bboxes = utils.nms(bboxes, 0.45, method='nms')  # 这一步是 将刚刚提取出来的信息进行筛选，返回最好的预测值，同样是三类。
    image = utils.draw_bbox(original_image, bboxes) # 这一步是把结果画到新图上面。
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = Image.fromarray(image)
    image.show()
    image.save(output_path)  # 保存图片到本地

object_detect = object_detect(input_path="./road.jpg", output_path='./demo.jpg')
