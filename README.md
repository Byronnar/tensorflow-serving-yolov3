## Introduction

主要对原tensorflow-yolov3版本做了许多细节上的改进,训练了Visdrone2019数据集，如果觉得好用记得star一下哦，谢谢！下面是改进细节:

1 修改了网络结构，支持了tensorflow-serving部署,自己训练的数据集也可以在线部署,并给出了 docker+yolov3-api测试脚本

2 修改了ulits文件，优化了demo展示,可以支持中文展示,添加了字体

3 详细的中文注释,代码更加易读,添加了数据敏感性处理,一定程度避免index的错误

4 修改了训练代码，支持其他数据集使用预训练模型了，模型体积减小三分之二，图片视频demo展示完都支持保存到本地,十分容易操作

5 借鉴视频检测的原理,添加了批量图片测试脚本,速度特别快(跟处理视频每一帧一样的速度)

6 添加了易使用的Anchors生成脚本以及各步操作完整的操作流程


## Part 1. demo展示
1. Clone this repo
```bashrc
$ git clone https://github.com/byronnar/tensorflow-serving-yolov3.git
$ cd tensorflow-serving-yolov3
$ pip install -r requirements.txt
```

2. Load the pre-trained TF checkpoint(`yolov3_coco.ckpt`) and export a .pb file. The checkpoint is provided from the forked repo not from the YOLO author though.
```bashrc
$ cd checkpoint
$ wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
$ tar -xvf yolov3_coco.tar.gz
$ cd ..
$ python convert_weight.py
$ python freeze_graph.py
```

3. Then you will get the `.pb` file in the root path.,  and run the demo script
```bashrc
$ python image_demo_Chinese.py
$ python video_demo.py # if use camera, set video_path = 0
```
Chinese image:

![images](https://github.com/Byronnar/tensorflow-serving-yolov3/blob/master/readme_images/demo.jpg)

4. Load the checkpoint file and export the SaveModel object to the `savemodel` folder for TensorFlow serving
```bashrc
$ python save_model.py
```

5. 将产生的saved model文件夹里面的 `yolov3` 文件夹复制到 `tmp` 文件夹下面，再运行
```
$ docker run -p 8501:8501 --mount type=bind,source=/tmp/yolov3/,target=/models/yolov3 -e MODEL_NAME=yolov3 -t tensorflow/serving &
```

$ cd serving-yolov3
$ python yolov3_api.py

Api Results:
![images](https://github.com/Byronnar/tensorflow-serving-yolov3/blob/master/readme_images/api.png)

## Part 2. 详细训练过程
### 2.1 Two files are required as follows:
# 先通过聚类算法产生anchors（也可以采用默认的anchors，除非你的数据集跟voc相差特别大） 
$ python anchors_generate.py

```
# 产生训练数据txt文件
$ python split.py
In train.txt
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# x_min, y_min etc. corresponds to the data in XML files
```

# 修改names文件
- [`class.names`]

```
person
bicycle
car
...
toothbrush
``` 

Train:
# 修改 config.py 文件，主要根据显存大小，注意batch_size，输入尺寸等参数

$ python train.py
$ python freeze_graph.py

Predict:
modify the image_demo.py
$ python image_demo.py

Visdrone results：
![visdrone](https://github.com/Byronnar/tensorflow-serving-yolov3/blob/master/readme_images/visdrone.jpg)

# 产生pb文件跟variables文件夹用于部署:
Using own datasets to deployment, you need first modify the yolov3.py line 47
$ python save_model.py


### 2.2 Download VOC dataset and prepare the above files
Download VOC PASCAL trainval and test data
```bashrc
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
Extract all of these tars into one directory and rename them, whi
