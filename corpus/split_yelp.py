import json
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
import re
import gzip
from keras.preprocessing import sequence
import numpy as np
import codecs
import operator
import datetime
import csv

in_fp = './Yelp/review.json'

limit = 99999999999
# top_k_words = 10000
min_reviews = 20
max_words = 1500

user_count = Counter()
item_count = Counter()
users = defaultdict(list)
items = defaultdict(list)

interactions = defaultdict(list)

print("Building count dictionary first to save memory...")
with codecs.open(in_fp, 'r', encoding = 'utf-8', errors = 'ignore') as f:
    for i, line in tqdm(enumerate(f), desc = "1st pass of reviews"):
        if(i>limit):
            break
        d = json.loads(line)
        user = d['user_id']
        item = d['business_id']
        user_count[user] += 1
        item_count[item] += 1

with codecs.open(in_fp, 'r', encoding = 'utf-8', errors = 'ignore') as f:
    for i,line in tqdm(enumerate(f), desc = "2nd pass of reviews"):
        if(i>limit):
            break
        d = json.loads(line)

        user = d['user_id']
        item = d['business_id']
        rating = d['stars']
        time = d['date']
        ts = int(datetime.datetime.strptime(time,
                            '%Y-%m-%d').strftime("%s"))
        if(user_count[user] < min_reviews or item_count[item] < min_reviews):
            continue
        text = d['text'].rstrip()
        interactions[user].append([item, rating, ts, text])

print("Number of users={}".format(len(interactions)))
# Filter interactions 2nd time

new_interactions = defaultdict(list)
new_items = []
for key, value in interactions.items():
    if(len(value)<min_reviews):
        continue
    else:
        new_interactions[key] = value
        new_items += [x[0] for x in value]

print('Filtered Users={}'.format(len(new_interactions)))

new_items_dict = dict(Counter(new_items))
new_interactions2 = defaultdict(list)
new_items2 = []

for key, value in new_interactions.items():
    new_v = [x for x in value if new_items_dict[x[0]]>min_reviews]
    if(len(new_v)<min_reviews):
        continue
    else:
        new_interactions2[key] = new_v
        new_items2 += [x[0] for x in new_v]

num_items = len(list(set(new_items2)))

print("Filtered Users={}".format(len(new_interactions2)))
print("Final number of items={}".format(num_items))

interactions = new_interactions2

import random

train = defaultdict(list)
dev = defaultdict(list)
test = defaultdict(list)

def remove_text(d):
    return [x[:-1] for x in d]

def dict_to_list(user, item_list):
    output = []
    for item in item_list:
        # Without text
        _out = [user] + item[x:-1]
        output.append(_out)
    return output

user_repr = defaultdict(list)
item_repr = defaultdict(list)

interaction_list = []

train, dev, test = [], [],[]

# for withheld reviews
user_repr2 = defaultdict(list)
item_repr2 = defaultdict(list)


# make reviews
for user, items in tqdm(interactions.items(), desc='make interactions'):
    if(len(items)<2):
        continue
    sorted_items = sorted(items, key=operator.itemgetter(2))
    train +=  [[user,x[0],x[1],x[-1]]for x in sorted_items[:-2]]
    dev += [[user, sorted_items[-2][0], sorted_items[-2][1],
                    sorted_items[-2][-1]]]
    test += [[user, sorted_items[-1][0], sorted_items[-1][1],
                    sorted_items[-1][-1]]]

    # for item in items:
    #     # last item is itext
    #     _inter = [user, item[0], item[1], item[-1]]
    #     interaction_list.append(_inter)

# print("Collected {} interactions".format(len(interaction_list)))

from collections import Counter

def down_sample(data):
    labels = [x[2] for x in data]
    count = Counter(labels).most_common(5)
    # print(Counter(labels))
    print(count)
    max_samples = count[-1][1]
    print("Downsampling to least class={}".format(max_samples))
    unique_lbls = list(set(labels))
    final_data = []
    for u in unique_lbls:
        _d = [x for x in data if x[2]==u][:max_samples]
        final_data += _d
    random.shuffle(final_data)
    labels = [x[2] for x in final_data]
    print(Counter(labels))
    return final_data


for t in train:
    user_repr[t[0]].append(t[-1])
    item_repr[t[1]].append(t[-1])

for t in test:
    user_repr2[t[0]].append(t[-1])
    item_repr2[t[1]].append(t[-1])

for t in dev:
    user_repr2[t[0]].append(t[-1])
    item_repr2[t[1]].append(t[-1])

print(user_repr.items()[0])

train = [x[:-1] for x in train]
test = [x[:-1] for x in test]
dev = [x[:-1] for x in dev]

print("==========================")
print("Set Stats")
print(len(train))
print(len(dev))
print(len(test))
print("==========================")

# print(train[:10])
# print(dev[:10])

def write_interactions(fp, data, mode='json'):
    with open(fp, 'w+') as f:
        if(mode=='csv'):
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(data)
        else:
            json.dump(data, f, indent=4)

write_interactions('./Yelp/train.txt', train, mode='csv')
write_interactions('./Yelp/dev.txt', dev, mode='csv')
write_interactions('./Yelp/test.txt', test, mode='csv')

write_interactions('./Yelp/user_text.json', user_repr)
write_interactions('./Yelp/user_text2.json', user_repr2)
write_interactions('./Yelp/item_text.json', item_repr)

print("Finished running file..")
