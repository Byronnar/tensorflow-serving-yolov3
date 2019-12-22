## Introduction

主要对原tensorflow-yolov3版本做了许多细节上的改进,训练了Visdrone2019数据集，准确率在87%左右, 如果觉得好用记得star一下哦，谢谢！
步骤十分详细，特别适合新手入门serving端部署，有什么问题可以提issues,下面是改进细节:

1. 修改了网络结构，支持了tensorflow-serving部署,自己训练的数据集也可以在线部署,并给出了 docker+yolov3-api测试脚本

2. 修改了ulits文件，优化了demo展示,可以支持中文展示,添加了字体

3. 详细的中文注释,代码更加易读,添加了数据敏感性处理,一定程度避免index的错误

4. 修改了训练代码，支持其他数据集使用预训练模型了，模型体积减小二分之一(如果不用指数平滑，可以减小到200多M一个模型，减小三分之二），图片视频demo展    示,都支持保存到本地,十分容易操作

5. 借鉴视频检测的原理,添加了批量图片测试脚本,速度特别快(跟处理视频每一帧一样的速度)

6. 添加了易使用的Anchors生成脚本以及各步操作完整的操作流程

7. 添加了 Mobilenetv2 backbone， 支持训练模型， 模型大小70多M


## Part 1. demo展示
1. 下载这份代码(本算法暂时是在ubuntu1804系统上实现(windows上理论上也可以使用，去除了命令行参数)
```bashrc
$ git clone https://github.com/byronnar/tensorflow-serving-yolov3.git
$ cd tensorflow-serving-yolov3
$ pip install -r requirements.txt
```
注：也可以直接点击 项目右上角 Clone r download 直接下载项目。

2. 下载预训练模型放到 checkpoint文件夹里面

百度网盘链接:      
```
https://pan.baidu.com/s/1Il1ASJq0MN59GRXlgJGAIw
密码：vw9x
```

谷歌云盘链接:        
```
https://drive.google.com/open?id=1aVnosAJmZYn1QPGL0iJ7Dnd4PTAukSU4
```

再运行下列代码：

```bashrc
$ cd checkpoint
$ tar -xvf yolov3_coco.tar.gz
$ cd ..
$ python convert_weight.py
$ python freeze_graph.py
```


补充资源链接(已经转化好的模型,不需要运行python convert_weight.py):
```

百度网盘链接：https://pan.baidu.com/s/12y0vmvKtspWuNMfUHTbPpA                  
密码：6xa8
```

3. 转化成.pb文件，运行下列代码：
```bashrc
$ python image_demo_Chinese.py             # 中文显示
$ python image_demo.py                                # 英文显示
$ python video_demo.py # if use camera, set video_path = 0
```
中文检测结果展示:

![images](https://github.com/Byronnar/tensorflow-serving-yolov3/blob/master/readme_images/demo.jpg)

4. 转化成 可部署的 saved model格式
```bashrc
$ python save_model.py
```

5. 将产生的saved model文件夹里面的 `yolov3` 文件夹复制到 `tmp` 文件夹下面，再运行
```
$ docker run -p 8501:8501 --mount type=bind,source=/tmp/yolov3/,target=/models/yolov3 -e MODEL_NAME=yolov3 -t tensorflow/serving &


$ cd serving-yolov3

$ python yolov3_api.py
```
服务器调用结果展示:
![images](https://github.com/Byronnar/tensorflow-serving-yolov3/blob/master/readme_images/api.png)

6. 模型评估
修改 config文件里面 的 # TEST options 部分
```
$ python evaluate.py
$ cd mAP
$ python main.py -na
```
工业检测数据集mAP：

![tv_map](https://github.com/Byronnar/tensorflow-serving-yolov3/blob/master/readme_images/tv_mAP.png)


## Part 2. 详细训练过程
2.1先准备好数据集,做成VOC2007格式,再通过聚类算法产生anchors（也可以采用默认的anchors，除非你的数据集跟voc相差特别大, 数据集可以用官方VOC2007） 
```
$ python anchors_generate.py

```
2.2 产生训练数据txt文件
```
$ python split.py
```
 train.txt 里面应该像这样:
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
x_min, y_min etc. corresponds to the data in XML files


2.3 修改names文件
- [`class.names`]

```
person
bicycle
car
...
toothbrush
``` 

正式训练:
2.4 修改 config.py 文件，主要根据显存大小，注意batch_size，输入尺寸等参数
```
$ python train.py
$ python freeze_graph.py
```
如果使用 mobilenetv2 backbone ，请运行：
```
python train_mobilenetv2.py
```

2.5 预测,修改 路径等相关参数:
修改  image_demo.py等文件
```
$ python image_demo.py
```
Visdrone results：

![visdrone](https://github.com/Byronnar/tensorflow-serving-yolov3/blob/master/readme_images/visdrone.jpg)

2.6 产生pb文件跟variables文件夹用于部署:

1 Using own datasets to deployment, you need first modify the yolov3.py line 47
```
$ python save_model.py
```

2 将产生的saved model文件夹里面的 `yolov3` 文件夹复制到 `tmp` 文件夹下面，再运行
```
$ docker run -p 8501:8501 --mount type=bind,source=/tmp/yolov3/,target=/models/yolov3 -e MODEL_NAME=yolov3 -t tensorflow/serving &

$ cd serving-yolov3

$ python yolov3_api.py
```

##  接下来要做的:

1.写 Tensorflow tiny版本 提高实时性

2.写 转换成 tensorflow lite 的脚本

## Reference
[YunYang1994](https://github.com/YunYang1994/tensorflow-yolov3.git)
