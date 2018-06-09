from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

def build_parser():
    """ This function produces the default argparser.

    Note that NOT all args are actively used. I used a generic
    argparser for all my projects, so you will surely see something that
    is not used. The important arguments are provided in the main readme. 
    """
    parser = argparse.ArgumentParser()
    ps = parser.add_argument
    ps("--dataset", dest="dataset", type=str,
        default='A2_Amazon_Instant_Video', help="Which dataset?")
    ps("--rnn_type", dest="rnn_type", type=str, metavar='<str>',
        default='RAW_MSE_MPCN_FN_FM', help="Compositional model name")")
    ps("--ablation_type", dest="rnn_type", type=str, metavar='<str>',
        default='', help='support ablation commands')
    ps("--comp_layer", dest="comp_layer", type=str, metavar='<str>',
        default='MUL_SUB', help="")
    ps("--opt", dest="opt", type=str, metavar='<str>', default='Adam',
       help="Optimization algorithm)")
    ps("--emb_size", dest="emb_size", type=int, metavar='<int>',
       default=50, help="Embeddings dimension (default=50)")
    ps("--rnn_size", dest="rnn_size", type=int, metavar='<int>',
       default=50, help="model-specific dimension. (default=50)")
    ps("--proj_size", dest="proj_size", type=int, metavar='<int>',
       default=50, help="proj dimension. (default=50)")
    ps("--conv_size", dest="conv_size", type=int, metavar='<int>',
       default=135, help="Conv Size for QRNN (default=3)")
    ps("--use_lower", dest="use_lower", type=int, metavar='<int>',
       default=1, help="Use all lowercase")
    ps("--batch-size", dest="batch_size", type=int, metavar='<int>',
       default=128, help="Batch size (default=128)")
    ps("--allow_growth", dest="allow_growth", type=int, metavar='<int>',
      default=0, help="Allow Growth")
    ps("--patience", dest="patience", type=int, metavar='<int>',
       default=3, help="Patience for halving LR")
    ps("--dev_lr", dest='dev_lr', type=int,
       metavar='<int>', default=0, help="Dev Learning Rate")
    ps("--rnn_layers", dest="rnn_layers", type=int,
       metavar='<int>', default=1, help="Number of RNN layers")
    ps("--decay_epoch", dest="decay_epoch", type=int,
       metavar='<int>', default=0, help="Decay everywhere n epochs")
    ps("--num_dense", dest="num_dense", type=int,
       metavar='<int>', default=0, help="Number of dense layers")
    ps("--num_proj", dest="num_proj", type=int, metavar='<int>',
       default=1, help="Number of projection layers")
    ps("--clip_output", dest="clip_output", type=int, metavar='<int>',
        default=0, help="clip output")
    ps("--factor", dest="factor", type=int, metavar='<int>',
       default=10, help="Number of factors (for FM model)")
    ps("--dropout", dest="dropout", type=float, metavar='<float>',
        default=0.8, help="The dropout probability.")
    ps("--rnn_dropout", dest="rnn_dropout", type=float, metavar='<float>',
        default=0.8, help="The dropout probability.")
    ps("--cell_dropout", dest="cell_dropout", type=float, metavar='<float>',
        default=1.0, help="The dropout probability.")
    ps("--emb_dropout", dest="emb_dropout", type=float, metavar='<float>',
        default=0.8, help="The dropout probability.")
    ps("--pretrained", dest="pretrained", type=int, metavar='<int>',
       default=0, help="Whether to use pretrained embeddings or not")
    ps("--epochs", dest="epochs", type=int, metavar='<int>',
       default=50, help="Number of epochs (default=50)")
    ps('--gpu', dest='gpu', type=int, metavar='<int>',
       default=0, help="Specify which GPU to use (default=0)")
    ps("--hdim", dest='hdim', type=int, metavar='<int>',
       default=50, help="Hidden layer size (default=50)")
    ps("--lr", dest='learn_rate', type=float,
       metavar='<float>', default=1E-3, help="Learning Rate")
    ps("--margin", dest='margin', type=float,
       metavar='<float>', default=0.2, help="Margin for hinge loss")
    ps("--clip_norm", dest='clip_norm', type=int,
       metavar='<int>', default=1, help="Clip Norm value for gradients")
    ps("--clip_embed", dest='clip_embed', type=int,
       metavar='<int>', default=0, help="Clip Norm value")
    ps("--trainable", dest='trainable', type=int, metavar='<int>',
       default=1, help="Trainable Word Embeddings (0|1)")
    ps('--l2_reg', dest='l2_reg', type=float, metavar='<float>',
       default=1E-6, help='L2 regularization, default=4E-6')
    ps('--eval', dest='eval', type=int, metavar='<int>',
       default=1, help='Epoch to evaluate results (default=1)')
    ps('--log', dest='log', type=int, metavar='<int>',
       default=1, help='1 to output to file and 0 otherwise')
    ps('--dev', dest='dev', type=int, metavar='<int>',
       default=1, help='1 for development set 0 to train-all')
    ps('--seed', dest='seed', type=int, default=1337, help='random seed (not used)')
    ps('--num_heads', dest='num_heads', type=int, default=1, help='number of heads')
    ps("--hard", dest="hard", type=int, metavar='<int>',
       default=1, help="Use hard att when using gumbel")
    ps('--toy', action='store_true', help='Use toy dataset (for fast testing, not supported)')
    ps('--tensorboard', action='store_true', help='To use tensorboard or not (may not work)')
    ps('--early_stop',  dest='early_stop', type=int,
       metavar='<int>', default=5, help='early stopping')
    ps('--wiggle_lr',  dest='wiggle_lr', type=float,
       metavar='<float>', default=1E-5, help='Wiggle lr')
    ps('--wiggle_after',  dest='wiggle_after', type=int,
       metavar='<int>', default=0, help='Wiggle lr')
    ps('--wiggle_score',  dest='wiggle_score', type=float,
       metavar='<float>', default=0.0, help='Wiggle score')
    ps('--translate_proj', dest='translate_proj', type=int,
       metavar='<int>', default=1, help='To translate project or not')
    ps('--test_bsz', dest='test_bsz', type=int,
       metavar='<int>', default=4, help='Multiplier for eval bsz')
    ps('--eval_train', dest='eval_train', type=int,
       metavar='<int>', default=1, help='To eval on train set or not')
    ps('--final_layer', dest='final_layer', type=int,
       metavar='<int>', default=1, help='To use final layer or not')
    ps('--data_link', dest='data_link', type=str, default='',
        help='data link')
    ps('--att_type', dest='att_type', type=str, default='SOFT',
        help='attention type')
    ps('--att_pool', dest='att_pool', type=str, default='MAX',
        help='pooling type for attention')
    ps('--num_class', dest='num_class', type=int,
       default=2, help='self explainatory..(not used for recommendation)')
    ps('--all_dropout', action='store_true',
       default=False, help='to dropout the embedding layer or not')
    ps("--qmax", dest="qmax", type=int, metavar='<int>',
       default=20, help="Max Length of Question (not used in rec)")
    ps("--char_max", dest="char_max", type=int, metavar='<int>',
       default=8, help="Max length of characters")
    ps("--amax", dest="amax", type=int, metavar='<int>',
       default=40, help="Max Length for Answer (not used in rec)")
    ps("--smax", dest="smax", type=int, metavar='<int>',
       default=30, help="Max Length of Sentences (per review)")
    ps("--dmax", dest="dmax", type=int, metavar='<int>',
       default=20, help="Max Number of documents (or reviews)")
    ps("--burn", dest="burn", type=int, metavar='<int>',
       default=0, help="Burn in period..")
    ps("--num_neg", dest="num_neg", type=int, metavar='<int>',
       default=6, help="Number of negative samples for pairwise training")
    ps("--injection", dest="injection", type=int, metavar='<int>',
       default=0, help="For hyperparameter injection")
    ps('--constraint',  type=int, metavar='<int>',
       default=0, help='Constraint embeddings to unit ball')
    ps('--sampling_mode', dest='sampling_mode',
       default='Mix', help='Which sampling mode..(not used for recsys)')
    ps('--base_encoder', dest='base_encoder',
       default='GLOVE', help='BaseEncoder for hierarchical models')
    ps('--save_embed', action='store_true', default=False,
       help='Save embeddings for visualisation')
    ps('--default_len', dest="default_len", type=int, metavar='<int>',
       default=1, help="Use default len or not")
    ps('--sort_batch', dest="sort_batch", type=int, metavar='<int>',
       default=0, help="To use sort-batch optimization or not")
    ps("--init", dest="init", type=float,
       metavar='<float>', default=0.01, help="Init Params")
    ps("--temperature", dest="temperature", type=float,
      metavar='<float>', default=0.5, help="Temperature")
    ps("--num_intra_proj", dest="num_intra_proj", type=int,
       metavar='<int>', default=1, help="Number of intra projection layers")
    ps("--num_ap_proj", dest="num_ap_proj", type=int,
       metavar='<int>', default=1, help="Number of AP projection layers")
    ps("--num_inter_proj", dest="num_inter_proj", type=int,
       metavar='<int>', default=1, help="Number of inter projection layers")
    ps("--dist_bias", dest="dist_bias", type=int,
       metavar='<int>', default=0, help="To use distance bias for intra-att or not")
    ps("--num_com", dest="num_com", type=int,
       metavar='<int>', default=1, help="Number of compare layers")
    ps("--show_att", dest="show_att", type=int,
      metavar='<int>', default=0, help="Display Attention")
    ps("--write_qual", dest="write_qual", type=int,
        metavar='<int>', default=0, help="write qual")
    ps("--show_affinity", dest="show_affinity", type=int,
        metavar='<int>', default=0, help="Display Affinity Matrix")
    ps("--init_type", dest="init_type", type=str,
       metavar='<str>', default='xavier', help="Init Type")
    ps("--rnn_init_type", dest="rnn_init_type", type=str,
       metavar='<str>', default='same', help="Init Type")
    ps("--init_emb", dest="init_emb", type=float,
       metavar='<float>', default=0.01, help="Init Embeddings")
    ps("--decay_lr", dest="decay_lr", type=float,
       metavar='<float>', default=0, help="Decay Learning Rate")
    ps("--decay_steps", dest="decay_steps", type=float,
       metavar='<float>', default=0, help="Decay Steps (manual)")
    ps("--decay_stairs", dest="decay_stairs", type=float,
       metavar='<float>', default=1, help="To use staircase or not")
    ps('--supply_neg', action='store_true', default=False,
       help='Supply neg samples to training set each iter')
    ps('--emb_type', dest='emb_type', type=str,
       default='glove', help='embedding type')
    ps('--log_dir', dest='log_dir', type=str,
       default='logs', help='log directory')
    ps('--use_cudnn', dest='use_cudnn', type=int, default=0)
    ps('--use_cove', dest='use_cove', type=int, default=0)
    return parser
