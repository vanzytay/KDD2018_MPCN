from __future__ import division
import cPickle as pickle
import csv
import numpy as np
import sys
import os
import cPickle as pickle
from nltk.corpus import stopwords
import json
import gzip
from tqdm import tqdm
from collections import Counter
from collections import defaultdict


def batchify(data, i, bsz, max_sample):
    start = int(i * bsz)
    end = int(i * bsz) + bsz
    if(end>max_sample):
        end = max_sample
    data = data[start:end]
    return data

def dict_to_list(data_dict):
    data_list = []
    for key, value in tqdm(data_dict.items(),
                            desc='dict conversion'):
        for v in value:
            data_list.append([key, v[0], v[1]])
    return data_list

def dictToFile(dict,path):
    print "Writing to {}".format(path)
    with gzip.open(path, 'w') as f:
        f.write(json.dumps(dict))

def dictFromFileUnicode(path):
    '''
    Read js file:
    key ->  unicode keys
    string values -> unicode value
    '''
    print "Loading {}".format(path)
    with gzip.open(path, 'r') as f:
        return json.loads(f.read())

def load_pickle(fin):
    with open(fin,'r') as f:
        obj = pickle.load(f)
    return obj

def select_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    if(gpu>=0):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

def load_pickle(fin):
    with open(fin,'r') as f:
        obj = pickle.load(f)
    return obj

def load_set(fin):
    data = []
    with open(fin, 'r') as f:
        reader= csv.reader(f, delimiter='\t')
        for r in reader:
            data.append(r)
    return data

def length_stats(lengths, name=''):
    print("=====================================")
    print("Length Statistics for {}".format(name))
    print("Max={}".format(np.max(lengths)))
    print("Median={}".format(np.median(lengths)))
    print("Mean={}".format(np.mean(lengths)))
    print("Min={}".format(np.min(lengths)))

def show_stats(name, x):
    print("{} max={} mean={} min={}".format(name,
                                        np.max(x),
                                        np.mean(x),
                                        np.min(x)))

def print_args(args, path=None):
    if path:
        output_file = open(path, 'w')
    args.command = ' '.join(sys.argv)
    items = vars(args)
    if path:
        output_file.write('=============================================== \n')
    for key in sorted(items.keys(), key=lambda s: s.lower()):
        value = items[key]
        if not value:
            value = "None"
        if path is not None:
            output_file.write("  " + key + ": " + str(items[key]) + "\n")
    if path:
        output_file.write('=============================================== \n')
    if path:
        output_file.close()
    del args.command

def mkdir_p(path):
    if path == '':
        return
    try:
        os.makedirs(path)
    except:
        pass
