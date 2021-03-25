#! /usr/bin/env python
# coding=utf-8


from easydict import EasyDict as edict


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = "./data/classes/coco.names"       # 不同数据集需要修改
__C.YOLO.ANCHORS                = "./data/anchors/basline_anchors.txt"   # 训练也可用coco_anchors.txt
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "resize"
__C.YOLO.ORIGINAL_WEIGHT        = "./checkpoint/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT            = "./checkpoint/yolov3_coco_demo.ckpt"

# 定义 Mobilenetv2 backbone 参数
__C.YOLO.BACKBONE_MOBILE        = True  # 定义mobilenetV2参数
__C.YOLO.GT_PER_GRID            = 3

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = "./2007_train.txt"
__C.TRAIN.BATCH_SIZE            = 4             # 尽量大一些, 不然训练效果较差
__C.TRAIN.INPUT_SIZE            = [416, 512, 608]   # 多尺度, 根据自己的显存跟图片分辨率去设置, 32的倍数即可
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 10
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30
__C.TRAIN.INITIAL_WEIGHT        = "./checkpoint/yolov3_coco_demo.ckpt"



# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = "./2007_test.txt"
__C.TEST.BATCH_SIZE             = 2
__C.TEST.INPUT_SIZE             = 544
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = "./checkpoint/yolov3_mobilenetv2.ckpt"
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.3
__C.TEST.IOU_THRESHOLD          = 0.45






