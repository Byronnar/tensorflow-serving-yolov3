# coding: utf-8
from __future__ import division
import xml.etree.ElementTree as ET
import os

names_dict = {}
cnt = 0
# -*- coding：utf-8 -*-
# 数据集划分,训练集，测试集，验证集

import xml.etree.ElementTree as ET

# 数据集划分,训练集，测试集，验证集
import random
import urllib.request
import os, tarfile

saveBasePath = r"./VOC2007/ImageSets"              # txt文件保存目录
total_xml = os.listdir(r'./VOC2007/Annotations')   # 获取标注文件（file_name.xml）

# 划分数据集为（训练，验证，测试集 = 49%,20%,30%）

val_percent = 0.2                              # 传参
test_percent = 0.1
trainval_percent = 0.9

# print(trainval_percent)

tv = int(len(total_xml) * trainval_percent)
#tr = int(len(total_xml) * train_percent)
ta = int(tv * val_percent)
tr = int(tv -ta)
tt = int(len(total_xml) * test_percent)

# 打乱训练文件（洗牌）
trainval = random.sample(range(len(total_xml)), tv)
train = random.sample(trainval, tr)

print("train size", tr)
print("val size", ta)
print("Test size", tt)

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

f = open( 'data/classes/visdrone.names', 'r' ).readlines()

for line in f:
    line = line.strip()
    names_dict[line] = cnt
    cnt += 1

voc_07 = './VOC2007'
# voc_12 = '/data/VOCdevkit/VOC2012'

anno_path = [os.path.join(voc_07, 'Annotations')]
img_path = [os.path.join(voc_07, 'JPEGImages')]

trainval_path = [os.path.join(voc_07, 'ImageSets/Main/trainval.txt')]
test_path = [os.path.join(voc_07, 'ImageSets/Main/test.txt')]

def parse_xml(path):
    tree = ET.parse(path)
    img_name = path.split('/')[-1][:-4]

    height = tree.findtext("./size/height")
    width = tree.findtext("./size/width")

    objects = [img_name, width, height]

    for obj in tree.findall('object'):
        difficult = obj.find('difficult').text
        if difficult == '1':
            continue
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text

        name = str(names_dict[name])
        objects.extend([name, xmin, ymin, xmax, ymax])
    if len(objects) > 1:
        return objects
    else:
        return None

test_cnt = 0
def gen_test_txt(txt_path):
    global test_cnt
    f = open(txt_path, 'w')

    for i, path in enumerate(test_path):
        img_names = open(path, 'r').readlines()
        for img_name in img_names:
            img_name = img_name.strip()
            xml_path = anno_path[i] + '/' + img_name + '.xml'
            objects = parse_xml(xml_path)
            if objects:
                objects[0] = img_path[i] + '/' + img_name + '.jpg'
                #print(objects[0])
                if os.path.exists(objects[0]):
                    objects.insert(0, str(test_cnt))
                    test_cnt += 1
                    objects = ' '.join(objects) + '\n'
                    #print(objects)
                    f.write(objects)
    f.close()


train_cnt = 0
def gen_train_txt(txt_path):
    global train_cnt
    f = open(txt_path, 'w')

    for i, path in enumerate(trainval_path):
        img_names = open(path, 'r').readlines()
        for img_name in img_names:
            img_name = img_name.strip()
            xml_path = anno_path[i] + '/' + img_name + '.xml'
            objects = parse_xml(xml_path)
            if objects:
                objects[0] = img_path[i] + '/' + img_name + '.jpg'
                if os.path.exists(objects[0]):
                    objects.insert(0, str(train_cnt))
                    train_cnt += 1
                    objects = ' '.join(objects) + '\n'
                    f.write(objects)
    f.close()


gen_train_txt('./train.txt')
gen_test_txt('./val.txt')
