#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .seq_op import *


def build_raw_cnn(embed, num_filters, filter_sizes=3,
                initializer=None, padding='VALID',
                dropout=None, name='', reuse=None):
    """ Builds a Convolutional Encoder and returns hidden outputs instead

    If num_digits of filter_sizes >1 then splits into list.
    e.g., 345 -> [3,4,5]. This assumes no filter is >=10 which is
    a reasonable assumption

    Args:
        embed: `tensor` input embedding of shape bsz x time_steps x dim
        num_filters: `int` cnn filters
        filter_sizes: `int` - explaination above
        initializer: tensorflow initializer
        dropout: tensorfow dropout placeholder
        reuse: to reuse weights or not

    Returns:
        pooled_outputs: `tensor` output embedding of shape
            [bsz x num_filters]

    """

    # convert filter_sizes to int
    filter_sizes = [int(x) for x in str(filter_sizes)]

   # print(filter_sizes)
    # dim = tf.shape(embed)[2]
    dim = embed.get_shape().as_list()[2]
    embed_expanded = tf.expand_dims(embed, -1)
    # seq_len = tf.shape(embed)[1]

    seq_len = embed.get_shape().as_list()[1]
    # seq_len = tf.shape(embed)[1]

    if(int(num_filters) % len(filter_sizes)>0):
        raise Exception("Warning: Filter Size is not completely divisible")
    _num_filters = int(num_filters / len(filter_sizes))
    outputs = []
    hidden_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        var_name = "raw_conv_layer_{}_{}".format(filter_size, name)
        with tf.variable_scope(var_name, reuse=reuse) as scope:
            filter_shape = [filter_size, dim, _num_filters]
            W1 = tf.get_variable("weights", filter_shape,
                                    initializer=initializer)
            b1 = tf.get_variable("bias", [_num_filters],
                        initializer=tf.constant_initializer([0.1]))
            conv =  tf.nn.conv1d(embed, W1, stride=1,
                            padding=padding, data_format="NHWC")

            h = tf.nn.relu(tf.nn.bias_add(conv, b1), name="relu")
            outputs.append(h)

    outputs = tf.concat(outputs, 2)
    # pooled_outputs = tf.reshape(pooled_outputs, [-1, num_filters])
    if(dropout is not None):
        outputs = tf.nn.dropout(outputs, dropout)

    return outputs

def build_cnn(embed, num_filters, filter_sizes=3, initializer=None,
                dropout=None, name='', reuse=None,
                round_filter=False):
    """ Builds a Convolutional Encoder

    If num_digits of filter_sizes >1 then splits into list.
    e.g., 345 -> [3,4,5]. This assumes no filter is >=10 which is
    a reasonable assumption

    Args:
        embed: `tensor` input embedding of shape bsz x time_steps x dim
        num_filters: `int` cnn filters
        filter_sizes: `int` - explaination above
        initializer: tensorflow initializer
        dropout: tensorfow dropout placeholder
        reuse: to reuse weights or not

    Returns:
        pooled_outputs: `tensor` output embedding of shape
            [bsz x num_filters]

    """

    # convert filter_sizes to int
    filter_sizes = [int(x) for x in str(filter_sizes)]

   # print(filter_sizes)
    # dim = tf.shape(embed)[2]
    dim = embed.get_shape().as_list()[2]
    embed_expanded = tf.expand_dims(embed, -1)
    # seq_len = tf.shape(embed)[1]

    seq_len = embed.get_shape().as_list()[1]
    # seq_len = tf.shape(embed)[1]

    if(int(num_filters) % len(filter_sizes)>0 and round_filter==False):
        raise Exception("Warning: Filter Size is not completely divisible")

    _num_filters = int(num_filters / len(filter_sizes))

    if(int(num_filters) % len(filter_sizes)>0):
        last_filter = int(num_filters) - int((len(filter_sizes)-1) * _num_filters)
        print("Num Filters per width={}".format(_num_filters))
        print("Last filter={}".format(last_filter))
    else:
        last_filter = _num_filters
    pooled_outputs = []
    hidden_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        var_name = "conv_layer_{}_{}".format(filter_size, name)
        with tf.variable_scope(var_name, reuse=reuse) as scope:
            if(i==(len(filter_sizes)-1)):
                print("Setting num_filters to {}".format(last_filter))
                _num_filters = last_filter
            filter_shape = [filter_size, dim, 1, _num_filters]
            W1 = tf.get_variable("weights", filter_shape,
                                    initializer=initializer)
            b1 = tf.get_variable("bias", [_num_filters],
                        initializer=tf.constant_initializer([0.1]))
            # conv =  tf.nn.conv1d(embed, W1, stride=1,
            #                 padding="VALID", data_format="NHWC")
            conv = tf.nn.conv2d(
                    embed_expanded,
                    W1,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b1), name="relu")
            pooled = tf.reduce_max(h, 1, keep_dims=True)
            # pooled = tf.nn.max_pool(
            #             h,
            #             ksize=[1, seq_len - filter_size + 1, 1, 1],
            #             strides=[1, 1, 1, 1],
            #             padding='VALID',
            #             name="pool")
            # print(pooled)
            pooled_outputs.append(pooled)

    pooled_outputs = tf.concat(pooled_outputs, 3)
    pooled_outputs = tf.reshape(pooled_outputs, [-1, num_filters])
    if(dropout is not None):
        pooled_outputs = tf.nn.dropout(pooled_outputs, dropout)

    return pooled_outputs
