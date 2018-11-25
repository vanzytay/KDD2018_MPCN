#!/usr/bin/env python
import csv
from nltk import word_tokenize
import random
import os
import numpy as np
from tqdm import tqdm
from collections import Counter
import cPickle as pickle
import string
import json
from collections import defaultdict
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from utilities import *

def flatten(l):
    return [item for sublist in l for item in sublist]

def load_reviews(fp):
    with open('./corpus/Yelp/{}.json'.format(fp),'r') as f:
        data = json.load(f)
    return data

def load_set(fp):
    data= []
    with open('./corpus/Yelp/{}.txt'.format(fp),'r') as f:
      reader = csv.reader(f, delimiter='\t')
      for r in reader:
        data.append(r)
    return data

def sent2words(sent):
    #print("==========================")
    #print(sent)
    sent = sent.splitlines()
    sent = ' '.join(sent)
    #_ sent =  sent.split(' ')
    _sent = tylib_tokenize(sent, setting='nltk')
    return _sent

def review2words(review):
    return [sent2words(x) for x in review]

def get_words(data_dict):
    data_list = data_dict.items()
    reviews = [flatten(review2words(x[1])) for x in data_list]
    words = []
    for r in tqdm(reviews, desc='parsing words'):
        words += r
    return words

user_text = load_reviews('user_text')
item_text = load_reviews('item_text')

user_text2 = load_reviews('user_text2')

print("Number of Users={}".format(len(user_text)))
print("Number of Items={}".format(len(item_text)))

user_ids = user_text.keys()
item_ids = item_text.keys()

user_index = {key:index for index, key in enumerate(user_ids)}
item_index = {key:index for index, key in enumerate(item_ids)}

user_text = {user_index[key]:value for key, value in user_text.items()}
item_text = {item_index[key]:value for key, value in item_text.items()}

user_text2 = {user_index[key]:value for key, value in user_text2.items()}


# Preprocessing text

def preprocess_dict(data_dict):
    words = []
    for key, value in tqdm(data_dict.items(),desc='preprocessing'):
        # print("=============================")
        # print(value)
        new_val = review2words(value)
        # print(new_val)
        raw_words = flatten(new_val)
        # print(raw_words)
        words += raw_words

        _str = [' '.join(x) for x in new_val]
        # print(_str)
        # for s in _str:
        #     print(s)
        data_dict[key] = _str
    return data_dict, words


user_text, words = preprocess_dict(user_text)
item_text, _ = preprocess_dict(item_text)
# print(user_text.items()[:10])
# print("Getting words...")
# user_words = get_words(user_text)

# item_words = get_words(item_text)

words = [x.lower() for x in words]

train  = load_set('train')
dev  = load_set('dev')
test  = load_set('test')

def process_set(d):
  try:
    return [[user_index[d[0]], item_index[d[1]], float(d[2])]]
  except:
    return []

train = [process_set(x) for x in train]
dev = [process_set(x) for x in dev]
test = [process_set(x) for x in test]

train =[x[0] for x in train if len(x)>0]
dev =[x[0] for x in dev if len(x)>0]
test =[x[0] for x in test if len(x)>0]

print('Train={} Dev={} Test={}'.format(len(train),len(dev),len(test)))

all_ratings = train + dev + test

user_dict = defaultdict(list)
rating_dict = {}
for t in tqdm(all_ratings, desc='rebuilding user dict'):
  user_dict[t[0]].append(t[1])
  rating_dict[str(tuple([t[0],t[1]]))] = t[2]


user_negative = {}
# make ranking dictionary
testing_users = [x[0] for x in test]
testing_users += [x[0] for x in dev]

testing_users = list(set(testing_users))
print("Number of unique testing users={}".format(len(testing_users)))

sample_count = 100

user_negative = {}

all_items = set([i for i in range(len(item_index))])

for user in testing_users:
  # Get ratings
  ui = set(user_dict[user])
  never_rated = list(all_items - ui)
  _sample_count = min(len(never_rated), sample_count)
  # print(len(never_rated))
  sampled = random.sample(never_rated, _sample_count)
  # print(sampled)
  sampled = [str(x) for x in sampled]
  user_negative[user] = ' '.join(sampled[:_sample_count])


print(user_negative.items()[:5])



print("Building Indexes")
word_index, index_word = build_word_index(words,
                              min_count=20,
                              extra_words=['<pad>','<unk>','<br>'],
                              lower=False)

words = list(set(words))
print(words[:50])

def repr_convert(repr_dict, word_index):
    def word2id(word):
        try:
            return word_index[word]
        except:
            return 1
    def sent2words(sent):
        sent = sent.split(' ')
        return ' '.join([str(word2id(x)) for x in sent])
    for key, value in tqdm(repr_dict.items(), desc='repr convert'):
        repr_dict[key] = [sent2words(x) for x in value]
    return repr_dict

user_text = repr_convert(user_text, word_index)
item_text = repr_convert(item_text, word_index)

user_text2 = repr_convert(user_text2, word_index)

print("Collecting Characters..")
chars = []
for t in tqdm(words, desc='Collecting Chars'):
    for c in t:
        chars += c


char_index, index_char = build_word_index(chars,
                            min_count=0,
                            extra_words=['<pad>','<unk>','<br>'],
                            lower=False)


print('Vocab size = {}'.format(len(word_index)))
print("Char Size ={}".format(len(char_index)))

fp = './datasets/yelp17/'

if not os.path.exists(fp):
    os.makedirs(fp)

build_embeddings(word_index, index_word, out_dir=fp,
  init_type='uniform', init_val=0.01,
  normalize=False, emb_types=[('glove',50)])

print("Saved Glove")

env = {
  'train':train,
  'dev':dev,
  'test':test,
  'user_text':user_text,
  'user_text2':user_text2,
  'item_text':item_text,
  'word_index':word_index,
  'char_index':char_index,
  'user_index':user_index,
  'item_index':item_index,
  'user_negative':user_negative
}

dictToFile(env,'{}env.gz'.format(fp))
