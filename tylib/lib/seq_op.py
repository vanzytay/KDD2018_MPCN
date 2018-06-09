from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

''' Sequence related ops

Such ops are useful for processing sequences. Inputs are typically
bsz x timesteps x dim long. And mostly output to bsz x dim,
applying a function across the temporal dimension.
'''

def clip_sentence(sentence, sizes):
    """ Clip the input sentence placeholders to the length of
        the longest one in the batch. This saves processing time.

    Args:
        sentence: `tensor`shape (batch, time_steps)
        sizes `tensor` shape (batch)

    Return:
        clipped_sent: `tensor` with shape (batch, time_steps)
    """

    max_batch_size = tf.reduce_max(sizes)
    clipped_sent = tf.slice(sentence, [0, 0],
                            tf.stack([-1, max_batch_size]))
    # clipped_sent = tf.reshape(clipped_sent, [-1, max_batch_size])
    return clipped_sent, max_batch_size

def mean_over_time(inputs, lengths):
    ''' Implements a MoT layer.

    Takes average vector across temporal dimension.

    Args:
        inputs: `tensor` [bsz x timestep x dim]
        lengths: `tensor` [bsz x 1] of sequence lengths

    Returns:
        mean_vec:`tensor` [bsz x dim]
    '''
    mean_vec = tf.reduce_sum(inputs, 1)
    mean_vec = tf.div(mean_vec, tf.cast(lengths, tf.float32))
    return mean_vec

def last_relevant(output, length):
    ''' Gets last relevant state from RNN based model

    Args:
        output: `tensor` RNN outputs [bsz x timestep x dim]
        length: `tensor` [bsz x 1] of sequence lengths

    Returns:
        relevant: `tensor` Output vector [bsz x dim]
    '''
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    print(index)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    print(relevant)
    return relevant
