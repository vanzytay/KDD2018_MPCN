import cPickle as pickle
import hickle
import numpy as np

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer

import tensorflow as tf

def load_pickle(fin):
	with open(fin,'r') as f:
		obj = hickle.load(f)
	return obj

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor
  (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def model_stats():
  print("============================================================")
  print("List of all Trainable Variables")
  tvars = tf.trainable_variables()
  all_params = []
  for idx, v in enumerate(tvars):
    print(" var {:3}: {:15} {}".format(idx, str(v.get_shape()), v.name))
    num_params = 1
    param_list = v.get_shape().as_list()
    if(len(param_list)>1):
      for p in param_list:
        if(p>0):
          num_params = num_params * int(p)
    else:
      all_params.append(param_list[0])
    all_params.append(num_params)
  print("Total number of trainable parameters {}".format(np.sum(all_params)))
