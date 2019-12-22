#! /usr/bin/env python
# coding=utf-8


import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common_mobilenetv2 as common
import core.backbone_mobilenetv2 as backbone
from core.config import cfg


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""

    def __init__(self, input_data, trainable):

        self.trainable = trainable
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class = len(self.classes)
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD

        self.mobile = cfg.YOLO.BACKBONE_MOBILE
        self.gt_per_grid = cfg.YOLO.GT_PER_GRID

        if self.mobile:
            try:
                self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_nework_mobile(input_data)
            except:
                raise NotImplementedError("Can not build up yolov3 network!")

            with tf.variable_scope('pred_sbbox'):
                self.pred_sbbox = self.decode_mobile(conv_output=self.conv_sbbox, num_classes=self.num_class,
                                                     stride=self.strides[0])

            with tf.variable_scope('pred_mbbox'):
                self.pred_mbbox = self.decode_mobile(conv_output=self.conv_mbbox, num_classes=self.num_class,
                                                     stride=self.strides[1])

            with tf.variable_scope('pred_lbbox'):
                self.pred_lbbox = self.decode_mobile(conv_output=self.conv_lbbox, num_classes=self.num_class,
                                                     stride=self.strides[2])


            """
            with tf.variable_scope('pred_multi_scale'):
                self.pred_multi_scale = tf.concat([tf.reshape(self.pred_sbbox, [-1, 85]),
                                                   tf.reshape(self.pred_mbbox, [-1, 85]),
                                                   tf.reshape(self.pred_lbbox, [-1, 85])], axis=0, name='concat')
            """
            # hand-coded the dimensions: if 608, use 19; if 416, use 13
            with tf.variable_scope('pred_multi_scale'):
                self.pred_multi_scale = tf.concat([tf.reshape(self.pred_sbbox, [-1, 19, 19, 85]),
                                                   tf.reshape(self.pred_mbbox, [-1, 19, 19, 85]),
                                                   tf.reshape(self.pred_lbbox, [-1, 19, 19, 85])], axis=0,
                                                  name='concat')
            # 说明,如果训练自己的数据集,将85改成 数据集里面的  类别数+5 ,再进行模型转化

    # 构建mobile网络
    def __build_nework_mobile(self, input_data):

        feature_map_s, feature_map_m, feature_map_l = backbone.MobilenetV2(input_data, self.trainable)

        conv = common.convolutional(name='conv0', input_data=feature_map_l, filters_shape=(1, 1, 1280, 512),
                                    trainable=self.trainable)
        conv = common.separable_conv(name='conv1', input_data=conv, input_c=512, output_c=1024,
                                     trainable=self.trainable)
        conv = common.convolutional(name='conv2', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                    trainable=self.trainable)
        conv = common.separable_conv(name='conv3', input_data=conv, input_c=512, output_c=1024,
                                     trainable=self.trainable)
        conv = common.convolutional(name='conv4', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                    trainable=self.trainable)

        # ----------**********---------- Detection branch of large object ----------**********----------
        conv_lbbox = common.separable_conv(name='conv5', input_data=conv, input_c=512, output_c=1024,
                                           trainable=self.trainable)
        conv_lbbox = common.convolutional(name='conv_lbbox', input_data=conv_lbbox,
                                          filters_shape=(1, 1, 1024, self.gt_per_grid * (self.num_class + 5)),
                                          trainable=self.trainable, downsample=False, activate=False, bn=False)
        # ----------**********---------- Detection branch of large object ----------**********----------

        # ----------**********---------- up sample and merge features map ----------**********----------
        conv = common.convolutional(name='conv7', input_data=conv, filters_shape=(1, 1, 512, 256),
                                    trainable=self.trainable)
        conv = common.upsample(name='upsample0', input_data=conv)
        conv = common.route(name='route0', previous_output=feature_map_m, current_output=conv)
        # ----------**********---------- up sample and merge features map ----------**********----------

        conv = common.convolutional(name='conv8', input_data=conv, filters_shape=(1, 1, 96 + 256, 256),
                                    trainable=self.trainable)
        conv = common.separable_conv('conv9', input_data=conv, input_c=256, output_c=512, trainable=self.trainable)
        conv = common.convolutional(name='conv10', input_data=conv, filters_shape=(1, 1, 512, 256),
                                    trainable=self.trainable)
        conv = common.separable_conv('conv11', input_data=conv, input_c=256, output_c=512, trainable=self.trainable)
        conv = common.convolutional(name='conv12', input_data=conv, filters_shape=(1, 1, 512, 256),
                                    trainable=self.trainable)

        # ----------**********---------- Detection branch of middle object ----------**********----------
        conv_mbbox = common.separable_conv(name='conv13', input_data=conv, input_c=256, output_c=512,
                                           trainable=self.trainable)
        conv_mbbox = common.convolutional(name='conv_mbbox', input_data=conv_mbbox,
                                          filters_shape=(1, 1, 512, self.gt_per_grid * (self.num_class + 5)),
                                          trainable=self.trainable, downsample=False, activate=False, bn=False)
        # ----------**********---------- Detection branch of middle object ----------**********----------

        # ----------**********---------- up sample and merge features map ----------**********----------
        conv = common.convolutional(name='conv15', input_data=conv, filters_shape=(1, 1, 256, 128),
                                    trainable=self.trainable)
        conv = common.upsample(name='upsample1', input_data=conv)
        conv = common.route(name='route1', previous_output=feature_map_s, current_output=conv)
        # ----------**********---------- up sample and merge features map ----------**********----------

        conv = common.convolutional(name='conv16', input_data=conv, filters_shape=(1, 1, 32 + 128, 128),
                                    trainable=self.trainable)
        conv = common.separable_conv(name='conv17', input_data=conv, input_c=128, output_c=256,
                                     trainable=self.trainable)
        conv = common.convolutional(name='conv18', input_data=conv, filters_shape=(1, 1, 256, 128),
                                    trainable=self.trainable)
        conv = common.separable_conv(name='conv19', input_data=conv, input_c=128, output_c=256,
                                     trainable=self.trainable)
        conv = common.convolutional(name='conv20', input_data=conv, filters_shape=(1, 1, 256, 128),
                                    trainable=self.trainable)

        # ----------**********---------- Detection branch of small object ----------**********----------
        conv_sbbox = common.separable_conv(name='conv21', input_data=conv, input_c=128, output_c=256,
                                           trainable=self.trainable)
        conv_sbbox = common.convolutional(name='conv_sbbox', input_data=conv_sbbox,
                                          filters_shape=(1, 1, 256, self.gt_per_grid * (self.num_class + 5)),
                                          trainable=self.trainable, downsample=False, activate=False, bn=False)
        # ----------**********---------- Detection branch of small object ----------**********----------

        return conv_lbbox, conv_mbbox, conv_sbbox


    def decode_mobile(self, conv_output, num_classes, stride):

        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        gt_per_grid = conv_shape[3] // (5 + num_classes)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, gt_per_grid, 5 + num_classes))
        conv_raw_dx1dy1 = conv_output[:, :, :, :, 0:2]
        conv_raw_dx2dy2 = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, gt_per_grid, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xymin = (xy_grid + 0.5 - tf.exp(conv_raw_dx1dy1)) * stride
        pred_xymax = (xy_grid + 0.5 + tf.exp(conv_raw_dx2dy2)) * stride
        pred_corner = tf.concat([pred_xymin, pred_xymax], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)

        pred_prob = tf.sigmoid(conv_raw_prob)

        pred_bbox = tf.concat([pred_corner, pred_conf, pred_prob], axis=-1)
        return pred_bbox

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss


    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / tf.maximum(union_area, 1e-12)  # 避免学习率设置高了，出现NAN的情况

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / tf.maximum(enclose_area, 1e-12)
        # 避免学习率设置高了，出现NAN的情况
        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / tf.maximum(union_area, 1e-12)  # 避免学习率设置高了，出现NAN的情况

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size

        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.gt_per_grid, 5 + self.num_class))

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            #
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors=self.anchors[0], stride=self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors=self.anchors[2], stride=self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss
