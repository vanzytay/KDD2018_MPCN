from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from .nn import *
from .cnn import *

def get_distance_biases(time_steps, reuse_weights=False, dist_bias=10):
    """ Return a 2-d tensor with the values of the distance biases to be applied
    on the intra-attention matrix of size sentence_size

    This is for intra-attention

    Args:
        time_steps: `tensor` scalar

    Returns:
         2-d `tensor` (time_steps, time_steps)
    """
    with tf.variable_scope('distance-bias', reuse=reuse_weights):
        # this is d_{i-j}
        distance_bias = tf.get_variable('dist_bias', [dist_bias],
                                        initializer=tf.zeros_initializer())

        # messy tensor manipulation for indexing the biases
        r = tf.range(0, time_steps)
        r_matrix = tf.tile(tf.reshape(r, [1, -1]),
                           tf.stack([time_steps, 1]))
        raw_inds = r_matrix - tf.reshape(r, [-1, 1])
        clipped_inds = tf.clip_by_value(raw_inds, 0,
                                        dist_bias - 1)
        values = tf.nn.embedding_lookup(distance_bias, clipped_inds)
    return values

def intra_attention(sentence, dim, initializer=None, activation=None,
                    reuse=None, dist_bias=10, dropout=None,
                    weights_regularizer=None):
    ''' Computes intra-attention

    Follows IA model of https://arxiv.org/pdf/1606.01933.pdf

    Args:
        sentence: `tensor` [bsz x time_steps x dim]
        dim: `int` projected dimensions
        initializer: tensorflow initializer
        activation: tensorflow activation (i.e., tf.nn.relu)
        reuse: `bool` To reuse params or not
        dist_bias: `int` value of dist bias
        dropout: Tensorflow dropout placeholder
        weights_regularizer: Regularization for projection layers

    Returns:
        attended: `tensor [bsz x timesteps x (dim+original dim)]
        attention: attention vector

    '''
    with tf.variable_scope('intra_att') as scope:
        time_steps = tf.shape(sentence)[1]
        dist_biases = get_distance_biases(time_steps, dist_bias=dist_bias,
                                            reuse_weights=reuse)
        sentence = projection_layer(sentence,
                                    dim,
                                    name='intra_att_proj',
                                    activation=activation,
                                    weights_regularizer=weights_regularizer,
                                    initializer=initializer,
                                    dropout=dropout,
                                    use_fc=False,
                                    num_layers=2,
                                    reuse=reuse)
        sentence2 = tf.transpose(sentence, [0,2,1])
        raw_att = tf.matmul(sentence, sentence2)
        raw_att += dist_biases
        attention = matrix_softmax(raw_att)
        attended = tf.matmul(attention, sentence)
        return tf.concat([sentence, attended], 2), attention

def mask_3d(values, sentence_sizes, mask_value, dimension=2):
    """ Given a batch of matrices, each with shape m x n, mask the values in each
    row after the positions indicated in sentence_sizes.
    This function is supposed to mask the last columns in the raw attention
    matrix (e_{i, j}) in cases where the sentence2 is smaller than the
    maximum.

    Source https://github.com/erickrf/multiffn-nli/

    Args:
        values: `tensor` with shape (batch_size, m, n)
        sentence_sizes: `tensor` with shape (batch_size) containing the
            sentence sizes that should be limited
        mask_value: `float` to assign to items after sentence size
        dimension: `int` over which dimension to mask values

    Returns
        A tensor with the same shape as `values`
    """
    if dimension == 1:
        values = tf.transpose(values, [0, 2, 1])
    time_steps1 = tf.shape(values)[1]
    time_steps2 = tf.shape(values)[2]

    ones = tf.ones_like(values, dtype=tf.int32)
    pad_values = mask_value * tf.cast(ones, tf.float32)
    mask = tf.sequence_mask(sentence_sizes, time_steps2)

    # mask is (batch_size, sentence2_size). we have to tile it for 3d
    mask3d = tf.expand_dims(mask, 1)
    mask3d = tf.tile(mask3d, (1, time_steps1, 1))
    mask3d = tf.cast(mask3d, tf.float32)

    masked = values * mask3d
    # masked = tf.where(mask3d, values, pad_values)

    if dimension == 1:
        masked = tf.transpose(masked, [0, 2, 1])

    return masked

def matrix_softmax(values):
    ''' Implements a matrix-styled softmax

    Args:
        values `tensor` [bsz x a_len, b_len]

    Returns:
        A tensor of the same shape
    '''
    original_shape = tf.shape(values)
    num_units = original_shape[2]
    reshaped = tf.reshape(values, tf.stack([-1, num_units]))
    softmaxed = tf.nn.softmax(reshaped)
    return tf.reshape(softmaxed, original_shape)

def softmax_mask(val, mask):
    return -1E-30 * (1 - tf.cast(mask, tf.float32)) + val

def co_attention(input_a, input_b, reuse=False, name='', att_type='SOFT',
                pooling='MEAN', k=10, mask_diag=False, kernel_initializer=None,
                dropout=None, activation=None, seq_lens=[], clipped=False,
                transform_layers=0, proj_activation=tf.nn.relu,
                dist_bias=0, gumbel=False, temp=0.5, hard=1,
                model_type="", mask_a=None, mask_b=None):
    ''' Implements a Co-Attention Mechanism

    Note: For self-attention, set input_a and input_b to be same tensor.

    input_a and input_b are embeddings of bsz x num_vec x dim. Co-attention
    forms a similarity matrix between a,b and by performing pooling of this
    affinity matrix, it learns to attend over a and b.

    There are two main configurations.
        1) att_type sets the choice of function
    for learning the affinity scores.
        2) pooling sets the choice of pooling function over the matrice.

    Matrix pooling is commonly used for intra-attention also known as alignment
    based pooling. It is also possible to not use masks for this, and by setting
    mask_a, mask_b to None, this will throw a warning. But may or may not affect
    performance. This is generally okay if sequences are of constant length.

    Transform layers optionally project the input embeddings, embedding-wise using
    a linear or nonlinear transform. Proj_activation parameter controls the activation.

    Gumbel option is supported for HARD attention. Setting gumbel=True activates it
    while temperature controls how "hard" the attention is.


    Args:
        input_a: `tensor`. Shape=[bsz x max_steps x dim]
        input_b: `tensor`. Shape=[bsz x max_steps x dim]
        reuse:  `bool`. To reuse weights or not
        name:   `str`. Variable name
        att_type: `str`. Supports 'BILINEAR','TENSOR','MLP' and 'MD'
        pooling: 'str'. supports "MEAN",'MAX' and 'SUM' pooling
        k:  `int`. For multi-dimensional. Num_slice tensor or hidden
            layer.
        mask_diag: `bool` Supports masking against diagonal for self-att
        kernel_initializer: `Initializer function
        dropout: `tensor` dropout placeholder (default is disabled)
        activation: Activation function
        seq_lens: `list of 2 tensors` actual seq_lens for
            input_a and input_b
        mask_a: mask for input a
        mask_b: mask for input_b

    Returns:
        final_a: `tensor` Weighted representation of input_a.
        final_b: `tensor` Weighted representation of input_b.
        max_row: `tensor` Row-based attention weights.
        max_col: `tensor` Col-based attention weights.
        y:  `tensor` Affinity matrix

    '''

    if(kernel_initializer is None):
        kernel_initializer = tf.random_uniform_initializer()

    if(len(input_a.get_shape().as_list())<=2):
        # expand dims
        input_a = tf.expand_dims(input_a, 2)
        input_b = tf.expand_dims(input_b, 2)
        readjust = True
    else:
        readjust = False

    # print(input_a)
    orig_a = input_a
    orig_b = input_b
    a_len = tf.shape(input_a)[1]
    b_len = tf.shape(input_b)[1]
    input_dim = tf.shape(input_a)[2]
    if(clipped):
        max_len = tf.reduce_max([tf.shape(input_a)[1],
                                tf.shape(input_b)[2]])
    else:
        max_len = a_len

    shape = input_a.get_shape().as_list()
    dim = shape[2]

    if(dist_bias>0):
        time_steps = tf.shape(input_a)[1]
        dist_biases = get_distance_biases(time_steps, dist_bias=dist_bias,
                                            reuse_weights=reuse)

    if(transform_layers>=1):
        input_a = projection_layer(input_a,
                                dim,
                                name='att_proj_{}'.format(name),
                                activation=proj_activation,
                                initializer=kernel_initializer,
                                dropout=None,
                                reuse=reuse,
                                num_layers=transform_layers,
                                use_mode='None')
        input_b = projection_layer(input_b,
                                dim,
                                name='att_proj_{}'.format(name),
                                activation=proj_activation,
                                reuse=True,
                                initializer=kernel_initializer,
                                dropout=None,
                                num_layers=transform_layers,
                                use_mode='None')
    if(att_type == 'BILINEAR'):
        # Bilinear Attention
        with tf.variable_scope('att_{}'.format(name), reuse=reuse) as f:
            weights_U = tf.get_variable("weights_U", [dim, dim],
                                        initializer=kernel_initializer)
        _a = tf.reshape(input_a, [-1, dim])
        z = tf.matmul(_a, weights_U)
        z = tf.reshape(z, [-1, a_len, dim])
        y = tf.matmul(z, tf.transpose(input_b, [0, 2, 1]))
    elif(att_type == 'TENSOR'):
        # Tensor based Co-Attention
        with tf.variable_scope('att_{}'.format(name),
                                reuse=reuse) as f:
            weights_U = tf.get_variable(
                    "weights_T", [dim, dim * k],
                    initializer=kernel_initializer)
            _a = tf.reshape(input_a, [-1, dim])
            z = tf.matmul(_a, weights_U)
            z = tf.reshape(z, [-1, a_len * k, dim])
            y = tf.matmul(z, tf.transpose(input_b, [0, 2, 1]))
            y = tf.reshape(y, [-1, a_len, b_len, k])
            y = tf.reduce_max(y, 3)
    elif(att_type=='SOFT'):
        # Soft match without parameters
        _b = tf.transpose(input_b, [0,2,1])
        z = tf.matmul(input_a, _b)
        y = z
    else:
        a_aug = tf.tile(input_a, [1, b_len, 1])
        b_aug = tf.tile(input_b, [1, a_len, 1])
        output = tf.concat([a_aug, b_aug], 2)
        if(att_type == 'MLP'):
            # MLP-based Attention
            sim = projection_layer(output, 1,
                                name='{}_co_att'.format(name),
                                reuse=reuse,
                                num_layers=1,
                                activation=None)
            y = tf.reshape(sim, [-1, a_len, b_len])
        elif(att_type == 'MD'):
            # Multi-dimensional Attention
            sim = projection_layer(output, k,
                                    name='co_att', reuse=reuse,
                                    activation=tf.nn.relu)
            feat = tf.reshape(sim, [-1, k])
            sim_matrix = tf.contrib.layers.fully_connected(
                                   inputs=feat,
                                   num_outputs=1,
                                   weights_initializer=kernel_initializer,
                                   activation_fn=None)
            y = tf.reshape(sim_matrix, [-1, a_len, b_len])

    if(activation is not None):
        y = activation(y)

    if(mask_diag):
        # Create mask to prevent matching against itself
        mask = tf.ones([a_len, b_len])
        mask = tf.matrix_set_diag(mask, tf.zeros([max_len]))
        y = y * mask

    if(dist_bias>0):
        print("Adding Distance Bias..")
        y += dist_biases
    if(pooling=='MATRIX'):
        # This is the alignment based attention
        # Note: This is not used in the MPCN model.
        # But you can use it if you want.
        _y = tf.transpose(y, [0,2,1])
        if(mask_a is not None and mask_b is not None):
            mask_b = tf.expand_dims(mask_b, 1)
            mask_a = tf.expand_dims(mask_a, 1)
            # bsz x 1 x b_len
            mask_a = tf.tile(mask_a, [1, b_len, 1])
            mask_b = tf.tile(mask_b, [1, a_len, 1])
            _y = softmax_mask(_y, mask_a)
            y = softmax_mask(y, mask_b)
        else:
            print("[Warning] Using Co-Attention without Mask!")
        att2 = tf.nn.softmax(_y)
        att1 = tf.nn.softmax(y)
        final_a = tf.matmul(att2, orig_a)
        final_b = tf.matmul(att1, orig_b)
        _a2 = att2
        _a1 = att
    else:
        if(pooling=='MAX'):
            att_row = tf.reduce_max(y, 1)
            att_col = tf.reduce_max(y, 2)
        elif(pooling=='MIN'):
            att_row = tf.reduce_min(y, 1)
            att_col = tf.reduce_min(y, 2)
        elif(pooling=='SUM'):
            att_row = tf.reduce_sum(y, 1)
            att_col = tf.reduce_sum(y, 2)
        elif(pooling=='MEAN'):
            att_row = tf.reduce_mean(y, 1)
            att_col = tf.reduce_mean(y, 2)

        # Get attention weights
        if(gumbel):
            att_row = gumbel_softmax(att_row, temp, hard=hard)
            att_col = gumbel_softmax(att_col, temp, hard=hard)
        else:
            att_row = tf.nn.softmax(att_row)
            att_col = tf.nn.softmax(att_col)
        _a2 = att_row
        _a1 = att_col

        att_col = tf.expand_dims(att_col, 2)
        att_row = tf.expand_dims(att_row, 2)

        # Weighted Representations
        final_a = att_col * input_a
        final_b = att_row * input_b

    y = tf.reshape(y, tf.stack([-1, a_len, b_len]))

    if(dropout is not None):
        final_a = tf.nn.dropout(final_a, dropout)
        final_b = tf.nn.dropout(final_b, dropout)

    if(readjust):
        final_a = tf.squeeze(final_a, 2)
        final_b = tf.squeeze(final_b, 2)

    return final_a, final_b, _a1, _a2, y

def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax(logits, temperature, hard=1):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    # print(gumbel_softmax_sample)
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)

    if hard==1:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y

def dual_attention(inputs, num_filters, filter_size=5, initializer=None, reuse=None,
                name='', dropout=None):
    """ Dual Attention Model
    This is for D-ATT model
    """
    local_repr = local_attention(inputs, num_filters,
                                initializer=initializer,
                                reuse=reuse)
    global_repr = build_cnn(inputs, num_filters, filter_sizes=234,
                    initializer=initializer, name='global{}'.format(name),
                    reuse=reuse, round_filter=True)
    lg = tf.concat([local_repr, global_repr], 1)
    final_repr = ffn(lg, num_filters, initializer, reuse=reuse, num_layers=2,
                    dropout=dropout, activation=tf.nn.relu)
    return final_repr

def local_attention(inputs, num_filters, filter_size=5, initializer=None, reuse=None,
                        name=''):
    """ Local Attention
    This is for D-ATT model
    """
    print("Local Attentional")
    weighted_inputs, _ = convolutional_attention(inputs, filter_size=filter_size,
                    initializer=initializer, name='local{}'.format(name),
                    reuse=reuse)
    conv_output = build_cnn(weighted_inputs, num_filters, filter_sizes=3,
                    initializer=initializer, name='local{}'.format(name),
                    reuse=reuse)
    print(conv_output)
    return conv_output

def convolutional_attention(inputs, filter_size=5, initializer=None,
                                reuse=None, name=''):
    """ sliding attention
    This is for D-ATT model
    """
    with tf.variable_scope('conv_att_{}'.format(name), reuse=reuse) as f:
        dim = inputs.get_shape().as_list()[2]
        filter_shape = filter_shape = [filter_size, dim, 1]
        W1 = tf.get_variable("weights", filter_shape,
                                initializer=initializer)
        b1 = tf.get_variable("bias", [1],
                    initializer=tf.constant_initializer([0.1]))
        conv =  tf.nn.conv1d(inputs, W1, stride=1,
                        padding="SAME", data_format="NHWC")
        # this should be bsz x seq_len x 1
        conv += b1
        att = tf.nn.sigmoid(conv)
        weighted_inputs = inputs * att
        return weighted_inputs, att

def attention(inputs, context=None, reuse=False, name='',
              kernel_initializer=None, dropout=None, gumbel=False,
              actual_len=None, temperature=1.0, hard=1, reuse2=None,
              return_raw=False):
    ''' Implements Vanilla Attention Mechanism

    TODO:Fix docs

    Context is for conditioned attentions (on last vector or topic
    vectors)

    Args:
        inputs: `tensor`. input seq of [bsz x timesteps x dim]
        context: `tensor`. input vector of [bsz x dim]
        reuse: `bool`. whether to reuse parameters
        kernel_initializer: intializer function
        dropout: tensor placeholder for dropout keep prob

    Returns:
        h_final: `tensor`. output representation [bsz x dim]
        att: `tensor`. vector of attention weights
    '''
    if(kernel_initializer is None):
        kernel_initializer = tf.random_uniform_initializer()

    shape = inputs.get_shape().as_list()
    dim = shape[2]
    # seq_len = shape[1]
    seq_len = tf.shape(inputs)[1]
    with tf.variable_scope('attention_{}'.format(name), reuse=reuse) as f:
        weights_Y = tf.get_variable(
            "weights_Y", [dim, dim], initializer=kernel_initializer,
                    validate_shape=False)
        weights_w = tf.get_variable(
            "weights_w", [dim, 1], initializer=kernel_initializer,
                    validate_shape=False)
        tmp_inputs = tf.reshape(inputs, [-1, dim])
        H = inputs
        Y = tf.matmul(tmp_inputs, weights_Y)
        Y = tf.reshape(Y, [-1, seq_len, dim])

    if(context is not None):
        # Add context for conditioned attention
        with tf.variable_scope('att_context_{}'.format(name), reuse=reuse2) as f:
            weights_h = tf.get_variable(
                "weights_h", [dim, dim], initializer=kernel_initializer)
            Wh = tf.expand_dims(context, 1)
            Wh = tf.tile(Wh, [1, seq_len, 1], name='tiled_state')
            Wh = tf.reshape(Wh, [-1, dim])
            HN = tf.matmul(Wh, weights_h)
            HN = tf.reshape(HN, [-1, seq_len, dim])
            Y = tf.add(Y, HN)

    Y = tf.tanh(Y, name='M_matrix')
    Y = tf.reshape(Y, [-1, dim])
    a = tf.matmul(Y, weights_w)
    a = tf.reshape(a, [-1, seq_len])

    if(actual_len is not None):
        a = mask_zeros_1(a, actual_len, seq_len, expand=False)
    if(gumbel):
        a = gumbel_softmax(a, temperature, hard=hard)
    else:
        a = tf.nn.softmax(a, name='attention_vector')

    att = tf.expand_dims(a, 2)

    r = tf.reduce_sum(inputs * att, 1)

    h_final = r
    if(context is not None):
        # Projection Layer
        with tf.variable_scope('att_context_{}'.format(name),
                                                reuse=reuse2) as f:
            weights_P = tf.get_variable(
                "weights_P", [dim, dim], initializer=kernel_initializer)
            weights_X = tf.get_variable(
                "weights_X", [dim, dim], initializer=kernel_initializer)
            Wr = tf.matmul(r, weights_P)
            Wx = tf.matmul(context, weights_X)
            h_final = tf.tanh(tf.add(Wr, Wx))

    return h_final, att
