# coding: utf-8
from __future__ import division, print_function
from __future__ import division
import xml.etree.ElementTree as ET
import random
import os

names_dict = {}
cnt = 0


saveBasePath = r"./VOC2007/ImageSets"              # txt文件保存目录
total_xml = os.listdir(r'./VOC2007/Annotations')   # 获取标注文件（file_name.xml）

# 划分数据集是为了读取图片等数据,以便于拿到聚类结果

trainval_percent = 1

# print(trainval_percent)

tv = int(len(total_xml) * trainval_percent)

trainval = random.sample(range(len(total_xml)), tv)

ftrainval = open(os.path.join(saveBasePath, 'Main/trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'Main/test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'Main/train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'Main/val.txt'), 'w')

for i in range(len(total_xml)):                # 遍历所有 file_name.xml 文件
    name = total_xml[i][:-4] + '\n'            # 获取 file_name
    if i in trainval:
        ftrainval.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

f = open('./data/classes/vis.names', 'r').readlines()

for line in f:
    line = line.strip()
    # print(line)
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
        # print(name)
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


#***************************************anchors*****************************************************************
import numpy as np

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 10 or np.count_nonzero(y == 0) > 10:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def parse_anno(annotation_path, target_size=None):
    anno = open(annotation_path, 'r')
    result = []
    for line in anno:
        s = line.strip().split(' ')
        img_w = int(s[2])
        img_h = int(s[3])
        s = s[4:]
        box_cnt = len(s) // 5
        for i in range(box_cnt):
            x_min, y_min, x_max, y_max = float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
            width = x_max - x_min
            height = y_max - y_min
            # assert width > 0
            # assert height > 0
            # use letterbox resize, i.e. keep the original aspect ratio
            # get k-means anchors on the resized target image size
            if target_size is not None:
                resize_ratio = min(target_size[0] / img_w, target_size[1] / img_h)
                width *= resize_ratio
                height *= resize_ratio
                result.append([width, height])
            # get k-means anchors on the original image size
            else:
                result.append([width, height])
    result = np.asarray(result)
    return result


def get_kmeans(anno, cluster_num=9):

    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou


if __name__ == '__main__':
    # target resize format: [width, height]
    # if target_resize is speficied[416,416], the anchors are on the resized image scale
    # if target_resize is set to None, the anchors are on the original image scale 5,8, 11,12, 10,22, 24,17, 19,34, 46,27, 35,52, 73,62, 132,123

    #target_size = [416, 416]           #   1,2, 2,4, 5,4, 4,8, 7,8, 12,7, 9,14, 19,16, 34,32

    target_size = [608, 608]            #   1,3, 4,4, 3,8, 9,6, 7,13, 17,10, 13,20, 27,23, 50,46
    annotation_path = "./train.txt"
    anno_result = parse_anno(annotation_path, target_size=target_size)
    anchors, ave_iou = get_kmeans(anno_result, 9)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)

