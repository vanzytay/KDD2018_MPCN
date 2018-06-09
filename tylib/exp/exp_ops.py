from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from tqdm import tqdm

''' Some utilities for experiments
'''

def optimize_batch(data, thresholds=[]):
    """ Batch optimizations
    """
    output = [[] for x in range(len(thresholds)+1)]

    for x in tqdm(data):
        assigned = False
        for i, t in enumerate(thresholds):
            lw, up = t[0], t[1]
            if(x[1]>lw and x[3]>lw and x[1]<up and x[3]<up):
                output[i].append(x)
                assigned = True
        if(assigned==False):
            output[-1].append(x)
    return output

def optimized_batch_shuffle(train_lists):
    train_data = []
    for t in train_lists:
        _t = t[:]
        random.shuffle(_t)
        train_data += _t
    return train_data

def flatten_list(l):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list

def make_batch(data, bsz, i):
    ''' Make batch

    Args:
        data: `list` list containing data
        bsz: `int` batch size
        epoch: `int` current epoch

    Returns:
        batch: `list` segmented list from original data
    '''
    start = int(i * bsz)
    end = int(i * bsz) + bsz
    batch = data[start:end]
    if(len(batch)==0):
        return None
    return batch

def pad_to_max(seq, seq_max, pad_token=0):
    ''' Pad Sequence to sequence max
    '''
    while(len(seq)<seq_max):
        seq.append(pad_token)
    return seq[:seq_max]

def prep_flat_data_dict(data_dict, smax, dmax):
    ''' Prepares flat data dictionary for ranking

    Data dict is in format {user:data} where data can be
    a list of sequences. This function converts data dict into
    appropriate format.

    Pads data dict to smax, and then records length information.

    Example:
        Output format is {user:
                            {
                                lengths:[1,2,3],
                                vals:[[29,32,32,34,53,21,0,0]
                                    }
                                    }

    Args:
        data `dict` dict-list input
        smax `int` maximum words per sequence
        dmax `int` maximum document per sample

    Returns:
        new_data_dict `dict
    '''

    print("Preparing Flat Data Dict..")
    new_data_dict = {}

    # Get max_len
    all_lengths = []
    for key, value in data_dict.items():
        cur_len = 0
        for v in value[:dmax]:
            cur_len += len(v)
        all_lengths.append(cur_len)

    max_len = np.max(all_lengths)
    print("Max len for flat data dict={}".format(max_len))


    for key, value in data_dict.items():
        new_value = []
        lengths = []
        for v in value[:dmax]:
            lengths.append(len(v))
            #  = pad_to_max(v, smax)
            new_value += v[:smax]
        lengths = pad_to_max(lengths, dmax)
        new_value = pad_to_max(new_value, max_len)
        new_data_dict[key] = {
                                'lengths':lengths,
                                'vals':new_value
                              }

    return new_data_dict, max_len

def prep_flat_data_list(data, smax, dmax, threshold=True,
                            add_delimiter=-1):
    all_data = []
    all_lengths = []
    sdmax = smax * dmax
    for doc in tqdm(data, desc='building F-dict'):
        new_data = []
        data_lengths = []
        for data_list in doc[:dmax]:
            # for each document
            sent_lens = len(data_list)
            if(sent_lens==0):
                continue
            if(threshold and sent_lens>smax):
                sent_lens = smax

            data_list = data_list[:smax-1]
            # _data_list = pad_to_max(data_list, smax)
            new_data += data_list
            if(add_delimiter>0):
                new_data.append(add_delimiter)
            # data_lengths.append(sent_lens)
        # print(new_data)
        all_lengths.append(len(new_data))
        new_data = pad_to_max(new_data, sdmax)
        all_data.append(new_data)
    return all_data, all_lengths


def prep_hierarchical_data_list(data, smax, dmax, threshold=True):
    """ Converts and pads hierarchical data
    """
    # print("Preparing Hiearchical Data list")

    # print(data[0])
    #print(data)

    all_data = []
    all_lengths = []
    for doc in tqdm(data, desc='building H-dict'):
        new_data = []
        data_lengths = []
        for data_list in doc:
            # for each document
            sent_lens = len(data_list)
            if(sent_lens==0):
                continue
            if(threshold and sent_lens>smax):
                sent_lens = smax

            _data_list = pad_to_max(data_list, smax)
            new_data.append(_data_list)
            data_lengths.append(sent_lens)
        new_data = pad_to_max(new_data, dmax,
                            pad_token=[0 for i in range(smax)])

        _new_data = []
        for nd in new_data:
            # flatten lists
            _new_data += nd

        data_lengths = pad_to_max(data_lengths, dmax, pad_token=1)
        all_data.append(_new_data)
        all_lengths.append(data_lengths)
    return all_data, all_lengths


def prep_hierachical_data_dict(data_dict, smax, dmax):
    ''' Prepares data dictionary for ranking

    Data dict is in format {user:data} where data can be
    a list of sequences. This function converts data dict into
    appropriate format.

    Pads data dict to smax, and then records length information.

    Example:
        Output format is {user:
                            {
                                lengths:[1,2,3],
                                vals:[[29,0,0],
                                    [30,1,0],
                                    [1,2,3]]
                                    }
                                    }

    Args:
        data `dict` dict-list input
        smax `int` maximum words per sequence
        dmax `int` maximum document per sample

    Returns:
        new_data_dict `dict
    '''

    print("Preparing Hierarchical Data Dict..")
    new_data_dict = {}

    for key, value in data_dict.items():
        new_value = []
        lengths = []
        for v in value[:dmax]:
            lengths.append(len(v))
            new_v = pad_to_max(v, smax)
            new_value += new_v

        lengths = pad_to_max(lengths, dmax)
        new_value = pad_to_max(new_value, smax * dmax)
        new_data_dict[key] = {
                                'lengths':lengths,
                                'vals':new_value
                              }

    return new_data_dict


def prepare_ranking_train_set(pairs, data, neg_rank, name='',
                            neg_sampling='local', num_neg=5,
                            verify_truth=False, shuffle=True,
                            train_mode='pairwise',
                            vec_dict=None, feats_only=False):
    ''' Prepares set for Ranking / Retrieval Problems

    includes negative sampling procedure
        - local samples from own neg List
        - global samples from entire global set

    Usage:
        This is only for training set

    Args:
        pairs: `list` actual data [[pid1, pid2]]
        data: `dict` maps qid -> actual data
        neg_rank: `dict` dictionary of negative samples
        name: `str` name of the set (for printing purposes)
        neg_sampling: `str` the mode of neg samping. supports
            'local' and 'global' and `none`
        num_neg: `int` number of negative samples per training loop
        verify_truth: `bool` whether to check if neg_samples is golden.
        train_mode: `str` whether to use pointwise or pairwise format

    Returns:
        output: `list` of format
            pairwise->(p1,p1_len, p2, p2_len, neg, neg_len)
            pintwise->(p1,p1_len, p2, p2_len, label [0 or 1])

    '''

    print("Preparing {} set for Ranking".format(name))

    output = []

    for p in tqdm(pairs):
        # get qid
        p0 = data[str(p[0])]
        p1 = data[str(p[1])]

        # print(p0)
        p0_text = p0['vals']
        p1_text = p1['vals']
        p0_len = p0['lengths']
        p1_len = p1['lengths']

        neg_samples = neg_rank[str(p[0])]

        pos_feat = []

        if(vec_dict is not None):
            # Use TF-IDF of BoW Features
            pos_feat += pairwise_tf_idf_features(vec_dict, p[0], p[1])

        if(neg_sampling is not None):
            neg_choosen = random.sample(neg_samples, num_neg)
        else:
            neg_choosen = neg_samples
        for n in neg_choosen:
            neg_feat = []
            n_data = data[str(n)]
            n_text = n_data['vals']
            n_len = n_data['lengths']
            if(train_mode=='pointwise'):
                # Support pointwise format
                output.append([p0_text, p0_len, n_text, n_len, 0])
            elif(train_mode=='pairwise'):
                if(feats_only):
                    _out = [[-1] for i in range(0,6)]
                else:
                    _out = [p0_text, p0_len, p1_text, p1_len,
                                    n_text, n_len]

                if(vec_dict is not None):
                    neg_feat += pairwise_tf_idf_features(vec_dict,
                                                            p[0], n)
                _out.append(pos_feat)
                _out.append(neg_feat)
                output.append(_out)
                # print(_out)

        if(train_mode=='pointwise'):
            # Support pointwise format
            output.append([p0_text, p0_len, p1_text, p1_len, 1])

    if(shuffle):
        random.shuffle(output)

    # feature_len = len(output[-1][-1])
    # print("Number of features={}".format(feature_len))

    return output

def pairwise_tf_idf_features(data_dict, a, b):
    """
    Construct pairwise features from single dictionaries
    """

    fa = data_dict[str(a)]
    fb = data_dict[str(b)]

    # Add tf-idf vectors together
    pw_feat = np.array([fa, fb])
    pw_feat = pw_feat.reshape([-1]).tolist()
    return pw_feat


def prepare_ranking_eval_set(pairs, data, neg_rank, name='',
                            eval_neg=100, train_mode='pairwise',
                            vec_dict=None, feats_only=False):
    """
    Prepares ranking evaluation set

    We take an approach of generating all possible pairs and
    mapping them to a pair_id list.

    During evaluation, we extract the scores for pairs.

    This allows for larger batch size for evaluation.

    Args:
        pairs: `list` actual data [[pid1, pid2]]
        data: `dict` maps qid -> actual data
        neg_rank: `dict` dictionary of negative samples
        name: `str` name of the set (for printing purposes)

    Returns:
        output: `list` of format (p1,p1_len, p2, p2_len, -1, -1)
        pairs_ids: `list` of [[p1,p2]...] as mapping to output

    """

    print("Preparing {} set for Ranking".format(name))
    output = []
    pair_ids = []
    for p in tqdm(pairs):
        # get qid
        p0 = data[str(p[0])]
        p1 = data[str(p[1])]

        # print(p0)
        p0_text = p0['vals']
        p1_text = p1['vals']
        p0_len = p0['lengths']
        p1_len = p1['lengths']

        neg_samples = neg_rank[str(p[0])][:eval_neg]

        pos_feat = []

        # [-1] marks not used slot
        # since we only use first 4 slots  of the network
        if(train_mode=='pairwise'):
            if(feats_only):
                _out = [[-1] for i in range(0,6)]
            else:
                _out = [p0_text, p0_len, p1_text, p1_len,
                            [-1],[-1]]
            if(vec_dict is not None):
                # Use TF-IDF of BoW Features
                pos_feat += pairwise_tf_idf_features(vec_dict, p[0], p[1])
            _out.append(pos_feat)
            _out.append([-1])
            output.append(_out)

        elif(train_mode=='pointwise'):
            output.append([p0_text, p0_len, p1_text, p1_len, 1])

        pair_ids.append(tuple([p[0],p[1]]))

        for n in neg_samples:
            neg_feat = []
            n_data = data[str(n)]
            n_text = n_data['vals']
            n_len = n_data['lengths']
            if(train_mode=='pairwise'):
                if(feats_only):
                    _out = [[-1] for i in range(0,6)]
                else:
                    _out = [p0_text, p0_len, n_text, n_len, [-1],[-1]]
                if(vec_dict is not None):
                    neg_feat += pairwise_tf_idf_features(vec_dict, p[0], n)
                _out.append(neg_feat)
                _out.append([-1])
                output.append(_out)
            elif(train_mode=='pointwise'):
                output.append([p0_text, p0_len, n_text, n_len, 0])
            pair_ids.append(tuple([p[0],n]))

    feature_len = len(output[-1][-2])
    return output, pair_ids, feature_len
