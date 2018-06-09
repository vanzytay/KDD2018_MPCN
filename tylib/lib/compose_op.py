from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

"""Implements composition ops such as:
"""

def dot_product(a, b):
    return tf.reduce_sum(a * b, 1, keep_dims=True)

def build_fm(input_vec, k=5, reuse=None, name='', initializer=None,
                reshape=False):
    """ Factorization Machine Layer
    """

    if(initializer is None):
        initializer = tf.random_normal_initializer(0, 0.1)

    if(reshape):
        # seq_lens = input_vec.get_shape().as_list()[1]
        seq_lens = tf.shape(input_vec)[1]
        _dims = input_vec.get_shape().as_list()[2]
        input_vec = tf.reshape(input_vec, [-1, _dims])

    with tf.variable_scope("fm_{}".format(name), reuse = reuse) as scope:
        fm_k = k
        fm_p = input_vec.get_shape().as_list()[1]

         # Global bias and weights for each feature
        fm_w0 = tf.get_variable("fm_w0", [1],
                    initializer=tf.constant_initializer([0]))
        fm_w = tf.get_variable("fm_w", [fm_p],
                    initializer = tf.constant_initializer([0]))

        # Interaction factors, randomly initialized
        fm_V = tf.get_variable("fm_V", [fm_k, fm_p],
                    initializer=initializer)

        fm_linear_terms = fm_w0 +  tf.matmul(input_vec,
                                     tf.expand_dims(fm_w, 1))

        fm_interactions_part1 = tf.matmul(input_vec, tf.transpose(fm_V))
        fm_interactions_part1 = tf.pow(fm_interactions_part1, 2)

        fm_interactions_part2 = tf.matmul(tf.pow(input_vec, 2),
                                        tf.transpose(tf.pow(fm_V, 2)))

        fm_interactions = fm_interactions_part1 - fm_interactions_part2

        latent_dim = fm_interactions
        fm_interactions = tf.reduce_sum(fm_interactions, 1, keep_dims = True)
        fm_interactions = tf.multiply(0.5, fm_interactions)
        fm_prediction = tf.add(fm_linear_terms, fm_interactions)
        if(reshape):
            fm_prediction = tf.reshape(fm_prediction, [-1, seq_lens, 1])
            latent_dim = tf.reshape(latent_dim, [-1, seq_lens, k])

        return fm_prediction, latent_dim

def reshaper(x):
    seq_lens = tf.shape(x)[1]
    _dims = x.get_shape().as_list()[2]
    x = tf.reshape(x, [-1, _dims])
    return x, seq_lens, _dims
