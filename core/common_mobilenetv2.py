#! /usr/bin/env python
# coding=utf-8


import tensorflow as tf


def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):
    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            # conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
            #                                      gamma_initializer=tf.ones_initializer(),
            #                                      moving_mean_initializer=tf.zeros_initializer(),
            #                                      moving_variance_initializer=tf.ones_initializer(), training=trainable)
            conv = group_normalization(input_data=conv, input_channel=filters_shape[-1])
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def group_normalization(input_data, input_channel, num_group=32, eps=1e-5):
    with tf.variable_scope('group_normalization'):
        input_shape = tf.shape(input_data)
        N = input_shape[0]
        H = input_shape[1]
        W = input_shape[2]
        C = input_channel
        assert (C % num_group) == 0
        input_data = tf.reshape(input_data, (N, H, W, num_group, C // num_group))
        axes = (1, 2, 4)
        mean = tf.reduce_mean(input_data, axis=axes, keep_dims=True)
        std = tf.sqrt(tf.reduce_mean(tf.pow(input_data - mean, 2), axis=axes, keep_dims=True) + eps)
        input_data = 1.0 * (input_data - mean) / std
        input_data = tf.reshape(input_data, (N, H, W, C))
        scale = tf.get_variable(name='scale', shape=C, dtype=tf.float32,
                                initializer=tf.ones_initializer, trainable=True)
        shift = tf.get_variable(name='shift', shape=C, dtype=tf.float32,
                                initializer=tf.zeros_initializer, trainable=True)
    return scale * input_data + shift


def inverted_residual(name, input_data, input_c, output_c, trainable, downsample=False, t=6):
    with tf.variable_scope(name):
        expand_c = t * input_c

        with tf.variable_scope('expand'):
            if t > 1:
                expand_weight = tf.get_variable(name='weights', dtype=tf.float32, trainable=True,
                                                shape=(1, 1, input_c, expand_c),
                                                initializer=tf.random_normal_initializer(stddev=0.01))
                expand_conv = tf.nn.conv2d(input=input_data, filter=expand_weight, strides=(1, 1, 1, 1), padding="SAME")
                expand_conv = batch_normalization(input_data=expand_conv, input_c=expand_c, trainable=trainable)
                expand_conv = tf.nn.relu6(expand_conv)
            else:
                expand_conv = input_data

        with tf.variable_scope('depthwise'):
            if downsample:
                pad_data = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
                expand_conv = tf.pad(expand_conv, pad_data, 'CONSTANT')
                strides = (1, 2, 2, 1)
                padding = 'VALID'
            else:
                strides = (1, 1, 1, 1)
                padding = "SAME"
            dwise_weight = tf.get_variable(name='depthwise_weights', dtype=tf.float32, trainable=True,
                                           shape=(3, 3, expand_c, 1),
                                           initializer=tf.random_normal_initializer(stddev=0.01))
            dwise_conv = tf.nn.depthwise_conv2d(input=expand_conv, filter=dwise_weight, strides=strides,
                                                padding=padding)
            dwise_conv = batch_normalization(input_data=dwise_conv, input_c=expand_c, trainable=trainable)
            dwise_conv = tf.nn.relu6(dwise_conv)

        with tf.variable_scope('project'):
            pwise_weight = tf.get_variable(name='weights', dtype=tf.float32, trainable=True,
                                           shape=(1, 1, expand_c, output_c),
                                           initializer=tf.random_normal_initializer(stddev=0.01))
            pwise_conv = tf.nn.conv2d(input=dwise_conv, filter=pwise_weight, strides=(1, 1, 1, 1), padding="SAME")
            pwise_conv = batch_normalization(input_data=pwise_conv, input_c=output_c, trainable=trainable)
        if downsample or pwise_conv.get_shape().as_list()[3] != input_data.get_shape().as_list()[3]:
            return pwise_conv
        else:
            return input_data + pwise_conv


def separable_conv(name, input_data, input_c, output_c, trainable, downsample=False):
    with tf.variable_scope(name):
        with tf.variable_scope('depthwise'):
            if downsample:
                pad_data = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
                input_data = tf.pad(input_data, pad_data, 'CONSTANT')
                strides = (1, 2, 2, 1)
                padding = 'VALID'
            else:
                strides = (1, 1, 1, 1)
                padding = "SAME"
            dwise_weight = tf.get_variable(name='depthwise_weights', dtype=tf.float32, trainable=True,
                                           shape=(3, 3, input_c, 1),
                                           initializer=tf.random_normal_initializer(stddev=0.01))
            dwise_conv = tf.nn.depthwise_conv2d(input=input_data, filter=dwise_weight, strides=strides, padding=padding)
            dwise_conv = batch_normalization(input_data=dwise_conv, input_c=input_c, trainable=trainable)
            dwise_conv = tf.nn.relu6(dwise_conv)

        with tf.variable_scope('pointwise'):
            pwise_weight = tf.get_variable(name='pointwise_weights', dtype=tf.float32, trainable=True,
                                           shape=(1, 1, input_c, output_c),
                                           initializer=tf.random_normal_initializer(stddev=0.01))
            pwise_conv = tf.nn.conv2d(input=dwise_conv, filter=pwise_weight, strides=(1, 1, 1, 1), padding="SAME")
            pwise_conv = batch_normalization(input_data=pwise_conv, input_c=output_c, trainable=trainable)
            pwise_conv = tf.nn.relu6(pwise_conv)
        return pwise_conv


def batch_normalization(input_data, input_c, trainable, decay=0.9):
    with tf.variable_scope('BatchNorm'):
        gamma = tf.get_variable(name='gamma', shape=input_c, dtype=tf.float32,
                                initializer=tf.ones_initializer, trainable=True)
        beta = tf.get_variable(name='beta', shape=input_c, dtype=tf.float32,
                               initializer=tf.zeros_initializer, trainable=True)
        moving_mean = tf.get_variable(name='moving_mean', shape=input_c, dtype=tf.float32,
                                      initializer=tf.zeros_initializer, trainable=False)
        moving_variance = tf.get_variable(name='moving_variance', shape=input_c, dtype=tf.float32,
                                          initializer=tf.ones_initializer, trainable=False)

        def mean_and_var_update():
            axes = (0, 1, 2)
            batch_mean = tf.reduce_mean(input_data, axis=axes)
            batch_var = tf.reduce_mean(tf.pow(input_data - batch_mean, 2), axis=axes)
            with tf.control_dependencies([tf.assign(moving_mean, moving_mean * decay + batch_mean * (1 - decay)),
                                          tf.assign(moving_variance,
                                                    moving_variance * decay + batch_var * (1 - decay))]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, variance = tf.cond(trainable, mean_and_var_update, lambda: (moving_mean, moving_variance))
        return tf.nn.batch_normalization(input_data, mean, variance, beta, gamma, 1e-05)


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):
    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1, filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output


def route(name, previous_output, current_output):
    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2, 2), kernel_initializer=tf.random_normal_initializer())

    return output



