# -*- coding: utf-8 -*-

import requests
import cv2.cv2 as cv2
import utils
import numpy as np
import grpc
import tensorflow as tf
import time
from PIL import Image
from apis import predict_pb2
from apis import prediction_service_pb2_grpc

class ServingBackend:
    server = ""
    model = ""
    version = ""

    def __init__(self, server: str, model: str, version: str):
        self.model = model
        self.version = version
        self.server = server

    def status(self):
        url = TFServingUtil.generateStatusURL( self )
        r = requests.get( url ).text

        return r

    def metadata(self):
        url = TFServingUtil.generateMetadataURL( self )
        r = requests.get( url ).text

        return r

class TFServingUtil():  # 获取Post请求的URL
    @staticmethod
    def generateStatusURL(backend: ServingBackend) -> str:
        url = "%s/%s/models/%s" % (backend.server, backend.version, backend.model)

        return url

    @staticmethod
    def generateMetadataURL(backend: ServingBackend) -> str:
        url = "%s/%s/models/%s/metadata" % (backend.server, backend.version, backend.model)

        return url

    @staticmethod
    def generatePredictURL(backend: ServingBackend) -> str:
        url = "%s/%s/models/%s:predict" % (backend.server, backend.version, backend.model)

        return url

class yolov3ServingBackend( ServingBackend ):
    def predict(self, img, classes, threshold, output_path):
        t1 = time.time()
        input_size = 608
        num_classes = len( classes )

        original_image = cv2.imread( img )
        original_image = cv2.cvtColor( original_image, cv2.COLOR_BGR2RGB )
        original_image_size = original_image.shape[:2]
        image_data = utils.image_preporcess( np.copy( original_image ), [input_size, input_size] )

        images_data = []
        images_data.append( image_data )
        images_data_np = np.array( images_data ).astype( np.float32 )

        tensor = tf.contrib.util.make_tensor_proto( images_data_np, shape=list( images_data_np.shape ) )  # 单张图片加载

        # image_data = [image_data.tolist()]
        t2 = time.time()
        # print('图片预处理时间:{}'.format(t2-t1))
        url = self.server

        options = [('grpc.max_send_message_length', 512 * 1024 * 1024),
                   ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        channel = grpc.insecure_channel( url, options=options )

        stub = prediction_service_pb2_grpc.PredictionServiceStub( channel )

        r = predict_pb2.PredictRequest()
        r.model_spec.name = self.model
        r.model_spec.signature_name = 'predict'
        r.inputs['input'].CopyFrom( tensor )

        # res = stub.Predict(r, 10.0)
        result_future = stub.Predict.future( r, 10.0 )  # 10 secs timeout
        res = result_future.result()

        t3 = time.time()
        # print('GRPC 预测时间:{}'.format(t3-t2))

        arr = tf.make_ndarray( res.outputs['output'] )
        # print("arr", arr)
        # print("output shape", arr.shape)

        pred_bbox = np.reshape( arr, (-1, 5 + num_classes) )
        bboxes = utils.postprocess_boxes( pred_bbox, original_image_size, input_size, threshold )
        t4 = time.time()
        # print(']图片 postprocess_boxes 处理时间:{}'.format(t4-t3))
        bboxes = utils.nms( bboxes, 0.45, method='nms' )
        t5 = time.time()
        # print('图片 nms 处理时间:{}'.format( t5-t4))
        image = utils.draw_bbox( original_image, bboxes, classes, True )

        t6 = time.time()
        # # print("图片draw_bbox处理时间:", t6-t5)

        # image_np = cv2.cvtColor( image, cv2.COLOR_RGB2BGR )  # 转换一下通道
        # cv2.imwrite(filepath, image_np)
        t7 = time.time()
        # print('图片写出处理时间:{}'.format( t7-t6))
        image = Image.fromarray( image )
        image.show()
        image.save( output_path )  # 保存图片到本地


if __name__ == "__main__":

    input_path = "./road.jpg"
    output_path = './demo_grpc.jpg'
    server = '0.0.0.0:8500'             # grpc服务端口
    model = 'yolov3'
    version = 'v1'
    threshold = 0.3  # 置信度
    classes = 'person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic light,fire hydrant,stop sign, parking meter, bench, bird,cat,dog,horse,sheep,cow,elephant,bear, zebra, giraffe, backpack, umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple, sandwich,orange,broccoli,carrot,hot dog,pizza, donut,cake,chair,sofa,pottedplant,bed,diningtable,toilet,tvmonitor,laptop,mouse,remote, keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush'
    # 类别，替换了数据集之后这里要改。
    classes = classes.split( "," )
    yolov3 = yolov3ServingBackend( server, model, version )
    image = yolov3.predict( input_path, classes, threshold, output_path )