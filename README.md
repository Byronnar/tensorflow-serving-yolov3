## Introduction

#### 本项目主要对原tensorflow-yolov3版本做了许多细节上的改进, 增加了TensorFlow-Serving工程部署, 训练了多个数据集，包括Visdrone2019, 安全帽等, 安全帽mAP在98%左右, 推理速度1080上608的尺寸大概25fps.
#### 此项目步骤十分详细，特别适合新手入门serving端部署，有什么问题可以提issues, 如果觉得好用记得star一下哦，谢谢！下面是改进细节:
    1. 修改了网络结构，支持了tensorflow-serving部署,自己训练的数据集也可以在线部署, 并给出了 docker+python_client测试脚本, 支持HTTP跟GRPC协议[ 0325 新增 ] 
    
    2. 修改了ulits文件，优化了demo展示,可以支持中文展示,添加了支持显示成中文的字体
    
    3. 详细的中文注释,代码更加易读,添加了数据敏感性处理, 一定程度避免index的错误
    
    4. 修改了训练代码，支持其他数据集使用预训练模型了，模型体积减小二分之一(如果不用指数平滑，可以减小到200多M一个模型，减小三分之二），图片视频demo展    示,都支持保存到本地,十分容易操作
    
    5. 借鉴视频检测的原理, 添加了批量图片测试脚本,速度特别快(跟处理视频每一帧一样的速度)
    
    6. 添加了易使用的Anchors生成脚本以及各步操作完整的操作流程
    
    7. 添加了 Mobilenetv2 backbone， 支持训练,预测,评估以及部署模型，模型大小70多M [ 0325 新增 ] 
    
    8. 增加 ONNX 导出 [ 0325 新增 ] 
    
    9. 增加 GRPC 远程过程调用 Serving 接口 [ 0325 新增 ] 
    
    10. 增加训练好的安全帽检测模型,数据集跟模型都在release中可下载 [ 0325 新增 ]

### Part 1. demo展示
#### 1. 下载这份代码(本算法暂时是在ubuntu1804系统上实现(windows上理论上也可以使用，去除了命令行参数)
```bashrc
$ git clone https://github.com/byronnar/tensorflow-serving-yolov3.git
$ cd tensorflow-serving-yolov3
$ pip install -r requirements.txt
```
#### 注：也可以直接点击 项目右上角 Clone r download 直接下载项目。

#### 2. 下载预训练模型放到 checkpoint文件夹里面
##### 百度网盘链接:      
```
https://pan.baidu.com/s/1Il1ASJq0MN59GRXlgJGAIw
密码：vw9x
```

##### 谷歌云盘链接:        
```
https://drive.google.com/open?id=1aVnosAJmZYn1QPGL0iJ7Dnd4PTAukSU4
```

##### 再运行下列代码, 解压缩, 然后将ckpt模型转化成固化模型, .pb格式：
```bashrc
$ cd checkpoint
$ tar -xvf yolov3_coco.tar.gz
$ cd ..
$ python convert_weight.py
$ python freeze_graph.py
```

##### 补充资源链接(已经转化好的模型,不需要运行python convert_weight.py):
```

百度网盘链接：https://pan.baidu.com/s/12y0vmvKtspWuNMfUHTbPpA                  
密码：6xa8
```

#### 3. 加载固化模型，进行预测, 运行下列代码：
```bashrc
$ python image_demo_Chinese.py             # 中文显示
$ python image_demo.py                   # 英文显示
$ python images_demo_batch.py             # 批量图片测试
$ python video_demo.py                   # video_path = 0 代表使用摄像头
```
##### 中文检测结果展示:

![images](https://github.com/Byronnar/tensorflow-serving-yolov3/blob/master/readme_images/demo.jpg)

#### 4. 转化成 可部署的 saved model格式
```bashrc
$ python save_model.py
```

#### 5. 将产生的saved model文件夹里面的 `yolov3` 文件夹复制到 `tmp` 文件夹下面，再运行
```
$ docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=/tmp/yolov3/,target=/models/yolov3 -e MODEL_NAME=yolov3 -t tensorflow/serving &
### 如果需要使用GPU, 请使用Tensorflow-serving-GPU镜像

$ cd yolov3_tfserving

HTTP 接口:
$ python http_client.py

GRPC 接口:
$ python grpc_client.py

```

服务器调用结果展示:
![images](https://github.com/Byronnar/tensorflow-serving-yolov3/blob/master/readme_images/api.png)

#### 6 将pb模型导出为ONNX
```
python -m tf2onnx.convert --input ./checkpoint/yolov3_coco_v3.pb --inputs input/input_data:0[1,416,416,3] --outputs pred_sbbox/concat_2:0,pred_sbbox/concat_2:0,pred_lbbox/concat_2:0 --output ./checkpoint/yolov3_coco_v3.onnx --verbose --fold_const --opset 11

```

##### 安全帽数据集mAP(模型已经开源)：
![tv_map](https://github.com/Byronnar/tensorflow-serving-yolov3/blob/master/readme_images/mAP.png)


### Part 2. 详细训练过程
#### 2.1先准备好数据集,做成VOC2007格式,再通过聚类算法产生anchors（也可以采用默认的anchors，除非你的数据集跟voc相差特别大, 数据集可以用官方VOC2007） 
```
$ python anchors_generate.py

```
#### 2.2 产生训练数据txt文件
```
$ python split.py

    train.txt 里面应该像这样:
        xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
        xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
    解读:
        image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
        x_min, y_min etc. corresponds to the data in XML files

```

##### 2.3 修改names文件
- [`class.names`]

```
person
bicycle
car
...
toothbrush
``` 

##### 2.4 正式训练: 修改 config.py 文件，主要根据显存大小，注意batch_size，输入尺寸等参数
```
$ python train.py
$ python train_mobilenetv2.py   # 注意anchors 最好使用coco_anchors
$ tensorboard --logdir ./data/log    # 查看损失等变化曲线

```

##### 如果想使用 mobilenetv2 backbone ，请运行：
```
$ python freeze_graph.py
$ python freeze_graph_mobilenetv2.py
```

##### 2.5 预测,修改 路径等相关参数:
###### 修改 image_demo.py, config.py等文件
```
$ python image_demo.py
$ python image_demo_mobilenetv2.py
```
###### 安全帽佩戴检测结果：

![visdrone](https://github.com/Byronnar/tensorflow-serving-yolov3/blob/master/readme_images/demo_helmet.jpg)

#### 2.6 模型评估,计算MAP
##### 修改 config文件里面 的 # TEST options 部分
```
$ python evaluate.py
$ python evaluate_mobilenetv2.py
$ cd mAP
$ python main.py -na
```

#### 2.7 产生pb文件跟variables文件夹用于部署:

##### 2.7.1 Using own datasets to deployment, you need first modify the yolov3.py line 47
```
$ python save_model.py
$ python save_model_mobilenetv2.py
```

##### 2.7.2 将产生的saved model文件夹里面的 `yolov3` 文件夹复制到 `tmp` 文件夹下面，再运行
```
$ docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=/tmp/yolov3/,target=/models/yolov3 -e MODEL_NAME=yolov3 -t tensorflow/serving &
### 如果需要使用GPU, 请使用Tensorflow-serving-GPU镜像

$ cd yolov3_tfserving

HTTP 接口:
$ python http_client.py

GRPC 接口:
$ python grpc_client.py

```

###  接下来要做的:
    1. Tensorflow-YOLOV4-TensorRT GPU加速部署
    
    2. YOLOV5-NCNN工程化 安卓端加速部署

### Reference
#### 感谢:
[YunYang1994](https://github.com/YunYang1994/tensorflow-yolov3.git)
