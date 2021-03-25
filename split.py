# -*- coding：utf-8 -*-
# 数据集划分,训练集，测试集，验证集
from __future__ import division
import xml.etree.ElementTree as ET
import random
import os

def base_txt():
    saveBasePath = r"./VOC2007/ImageSets"              # txt文件保存目录
    total_xml = os.listdir(r'./VOC2007/Annotations')   # 获取标注文件（file_name.xml）

    # 划分数据集为（训练，验证，测试集 = 49%,20%,30%）

    val_percent = 0.2                              # 可以自己修改
    test_percent = 0.2
    trainval_percent = 0.8

    # print(trainval_percent)

    tv = int(len(total_xml) * trainval_percent)
    #tr = int(len(total_xml) * train_percent)
    ta = int(tv * val_percent)
    tr = int(tv -ta)
    tt = int(len(total_xml) * test_percent)

    # 打乱训练文件（洗牌）
    trainval = random.sample(range(len(total_xml)), tv)
    train = random.sample(trainval, tr)

    print("训练集图片数量：", tr)
    print("验证集图片数量：", ta)
    print("测试集图片数量：", tt)

    # with open('/tmp/VOC2007/split.txt', 'w', encoding='utf-8') as f:
    #     f.write(str(val_percent))

    ftrainval = open(os.path.join(saveBasePath, 'Main/trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'Main/test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'Main/train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'Main/val.txt'), 'w')

    for i in range(len(total_xml)):                # 遍历所有 file_name.xml 文件
        name = total_xml[i][:-4] + '\n'            # 获取 file_name
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

base_txt()

fi = open('./data/classes/voc.names', 'r')      # 按文件夹里面的文件修改好
txt = fi.readlines()
voc_class = []
for w in txt:
    w = w.replace('\n', '')
    voc_class.append(w)
print('数据集里面的类别为： ', voc_class)
classes = voc_class

def convert_annotation(year, image_id, list_file):

    in_file = open('./VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = '.'

sets=[ ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# wd = getcwd()
for year, image_set in sets:
    image_ids = open('./VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
