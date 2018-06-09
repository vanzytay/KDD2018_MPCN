from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

import operator
import argparse
import json
import gzip
import pickle
from datetime import datetime
import os
import random
from tqdm import tqdm

from keras.preprocessing import sequence
from tensorflow.contrib.tensorboard.plugins import projector
from .utilities import *
import sys

from keras.utils import np_utils
from keras.preprocessing import sequence
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import csv
import sys
import string

from collections import defaultdict

reload(sys)
sys.setdefaultencoding('UTF8')

class Experiment(object):
    ''' Implements a base experiment class for TensorFLow

    Contains commonly used util functions.

    Extend this base Experiment class
    '''

    def __init__(self):
        self.uuid = datetime.now().strftime("%d_%H:%M:%S")
        self.eval_test = defaultdict(list)
        self.eval_train = defaultdict(list)
        self.eval_dev = defaultdict(list)
        self.eval_test2 = defaultdict(list)
        self.eval_dev2 = defaultdict(list)
        self.wiggle = False
        self.loggers = defaultdict(dict)

    def register_to_log(self, set_type, epoch, attr, val):
        if(attr not in self.loggers):
            self.loggers[attr] = {'train':defaultdict(dict),
                                    'Dev':defaultdict(dict),
                                    'Test':defaultdict(dict)}
        self.loggers[attr][set_type][epoch] = val

    def dump_all_logs(self):
        for key, value in self.loggers.items():
            for set_type, data in value.items():
                self.write_log_values(data, key, set_type)

    def write_log_values(self, data, attr, set_type):
        fp = self.out_dir +'./{}_{}.log'.format(attr, set_type)
        with open(fp, 'w+') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(data.items())

    def _build_char_index(self):
        all_chars = list(string.printable)
        self.char_index = {char:index+2 for index, char in enumerate(all_chars)}
        self.char_index['<pad>'] = 0
        self.char_index['<unk>'] = 1

    def _setup(self):
        ''' Full Setup Procedure
        '''
         # Make directory for log file and saving models
        self._make_dir()
        # Select GPU
        self._designate_gpu()

    def _setup_tf(self, load_embeddings=True):
        ''' Setups TensorFlow
        '''
        tf.reset_default_graph()
        _config_proto = tf.ConfigProto(
                        allow_soft_placement=True,
                        intra_op_parallelism_threads=8)
        _config_proto.gpu_options.allow_growth = bool(self.args.allow_growth)

        self.sess = tf.Session(graph=self.mdl.graph,
                                config=_config_proto)
        if(self.args.debugger==1):
            # This doesn't seem to work? :(
            self.sess = tf_debug.TensorBoardDebugWrapperSession(
                                    self.sess,
                                    'localhost:6064')
        with self.mdl.graph.as_default():
            tf.set_random_seed(self.args.seed)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self._build_writers()
            if(load_embeddings):
                self._load_embeddings()
            print("Finished loading embeddings....")

    def _designate_gpu(self):
        ''' Choose GPU to use
        '''
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if(self.args.gpu == '-1'):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            print("Selecting GPU no.{}".format(self.args.gpu))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def _make_dir(self):
        ''' Make log directories
        '''
        # self.model_name = self.args.rnn_type
        self.hyp_str = self.uuid + '_' + self.model_name
        if(self.args.log == 1):
            self.out_dir = './{}/{}/{}/{}/'.format(
                self.args.log_dir,
                self.args.dataset, self.model_name, self.uuid)
            # print(self.out_dir)
            mkdir_p(self.out_dir)
            self.mdl_path = self.out_dir + '/mdl.ckpt'
            self.path = self.out_dir + '/logs.txt'
            print_args(self.args, path=self.path)

    def write_to_file(self, txt):
        ''' A wrapper for printing to log and on CLI
        '''
        try:
            if(self.args.log == 1):
                with open(self.path, 'a+') as f:
                    f.write(txt + '\n')
        except:
            pass
        print(txt)

    def _load_embeddings(self):
        ''' Loads pre-trained embeddings
        '''
        if(self.args.pretrained == 1):
            embedding = np.load('./datasets/{}/emb_{}_{}.npy'.format(
                self.args.dataset, self.args.emb_type,
                self.args.emb_size))
            print(embedding)
            # print(np.sum(embeddings, axis=1))
            if(self.args.normalize_embed==1):
                print("Normalizing Embedding...")
                sum_embed = np.sum(embedding, axis=1)
                embedding = embedding / sum_embed[:, np.newaxis]

            # if(self.args.paginate>0):
            #     # Add paginate tokens
            #     concat = np.zeros((2, self.args.emb_size))
            #     embedding = np.concatenate((embedding, concat), axis=0)

            print("loaded embeddings")
            print(embedding.shape)
            feed_dict = {self.mdl.emb_placeholder: embedding}
            self.sess.run(self.mdl.embeddings_init, feed_dict=feed_dict)

    def _build_writers(self):
        ''' Build TensorBoard writers
        '''
        if(self.args.tensorboard):
            self.test_dir =  './tf_logs/{}/test_{}'.format(
                                            self.args.dataset,
                                            self.hyp_str)

            print("Building Tensorboard Writers..")
            self.train_writer = tf.summary.FileWriter(
                './tf_logs/{}/train_{}'.format(self.args.dataset,
                                                self.hyp_str),
                                                self.sess.graph)
            self.dev_writer = tf.summary.FileWriter(
                './tf_logs/{}/dev_{}'.format(self.args.dataset,
                                                self.hyp_str),
                                                self.sess.graph)
            self.test_writer = tf.summary.FileWriter(self.test_dir,
                                                 self.sess.graph)

    def _register_eval_score(self, epoch, eval_type, metric, val):
        """ Registers eval metrics to class
        """
        eval_obj = {
            'metric':metric,
            'val':val
        }

        if(eval_type.lower()=='dev'):
            self.eval_dev[epoch].append(eval_obj)
        elif(eval_type.lower()=='test'):
            self.eval_test[epoch].append(eval_obj)
        elif(eval_type.lower()=='train'):
            self.eval_train[epoch].append(eval_obj)
        elif(eval_type.lower()=='dev2'):
            self.eval_dev2[epoch].append(eval_obj)
        elif(eval_type.lower()=='test2'):
            self.eval_test2[epoch].append(eval_obj)

    def _show_metrics(self, epoch, eval_list, show_metrics, name):
        """ Shows and outputs metrics
        """
        # print("Eval Metrics for [{}]".format(name))
        get_last = eval_list[epoch]
        for metric in get_last:
            # print(metric)
            if(metric['metric'] in show_metrics):
                self.write_to_file("[{}] {}={}".format(name,
                                                    metric['metric'],
                                                    metric['val']))


    def _select_test_by_dev(self, epoch, eval_dev, eval_test,
                            no_test=False, lower_is_better=False,
                            name='', has_dev=True):
        """ Outputs best test score based on dev score
        """

        self.write_to_file("====================================")
        primary_metrics = []
        test_metrics = []
        if(lower_is_better):
            reverse=False
        else:
            reverse=True
        # print(eval_dev)

        if(has_dev==True):
            for key, value in eval_dev.items():
                _val = [x for x in value if x['metric']==self.eval_primary]
                if(len(_val)==0):
                    continue
                primary_metrics.append([key, _val[0]])

            sorted_metrics = sorted(primary_metrics,
                                    key=operator.itemgetter(1),
                                        reverse=reverse)
            cur_dev_score = primary_metrics[-1][1]['val']
            best_epoch = sorted_metrics[0][0]

            if(no_test):
                # For MNLI or no test set
                print("[{}] Best epoch={}".format(name, best_epoch))
                self._show_metrics(best_epoch, eval_dev,
                                    self.show_metrics, name='best')
                if(self.args.wiggle_score>0 and self.wiggle==False):
                    if(cur_dev_score>self.args.wiggle_score):
                        print("Cur Dev Score at {}".format(cur_dev_score))
                        print("Activating Wiggle-SGD mode")
                        self.wiggle=True
                return best_epoch, cur_dev_score
        else:
            best_epoch = -1

        for key, value in eval_test.items():
            _val = [x for x in value if x['metric']==self.eval_primary]
            if(len(_val)==0):
                continue
            test_metrics.append([key, _val[0]])

        # if(len(primary_metrics)==0):
        #     return False

        sorted_test = sorted(test_metrics, key=operator.itemgetter(1),
                                    reverse=reverse)

        max_epoch = sorted_test[0][0]

        self.write_to_file("Best epoch={}".format(best_epoch))
        self._show_metrics(best_epoch, eval_test,
                            self.show_metrics, name='best')
        self.write_to_file("Maxed epoch={}".format(max_epoch))
        self._show_metrics(max_epoch, eval_test,
                            self.show_metrics, name='max')
        if(self.args.early_stop>0):
            # Use early stopping
            if(epoch - best_epoch > self.args.early_stop):
                # print("Ended at early stop..")
                return True, max_epoch, best_epoch
        if(self.args.wiggle_after>0 and self.wiggle==False):
            # use SGD wiggling
            if(epoch - best_epoch > self.args.wiggle_after):
                print("Activating Wiggle-SGD mode")
                self.wiggle = True
        if(self.args.wiggle_score>0 and self.wiggle==False):
            if(cur_dev_score>self.args.wiggle_score):
                print("Cur Dev Score at {}".format(cur_dev_score))
                print("Activating Wiggle-SGD mode")
                self.wiggle=True

        return False, max_epoch, best_epoch


    def _print_model_stats(self):
        ''' Returns total number of trainable parameters in model
        '''
        with self.mdl.graph.as_default():
            self.write_to_file("========================================")
            self.write_to_file("List of all Trainable Variables")
            tvars = tf.trainable_variables()
            all_params = []
            for idx, v in enumerate(tvars):
                try:
                    self.write_to_file(" var {:3}: {:15} {}".format(idx,
                                                str(v.get_shape()),
                                                v.name))
                    num_params = 1
                    param_list = v.get_shape().as_list()
                    if(len(param_list)>1):
                        for p in param_list:
                            if(p>0):
                                num_params = num_params * int(p)
                    else:
                        all_params.append(param_list[0])
                    all_params.append(num_params)
                except:
                    # print(v)
                    pass
            num_params = np.sum(all_params)
            self.write_to_file("Total number of trainable params {}".format(
                                num_params))
