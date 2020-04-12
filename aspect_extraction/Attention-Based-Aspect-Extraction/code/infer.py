import os

import numpy as np
from keras.models import load_model
from tqdm import tqdm
from sklearn.metrics import classification_report
import keras.backend as K
from keras.preprocessing import sequence

import utils as U
import reader as dataset
from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin

######### Get hyper-params in order to rebuild the model architecture ###########
# The hyper parameters should be exactly the same as those used for training

parser = U.add_common_args()
args = parser.parse_args()


out_dir = os.path.join(args.out_dir_path , args.domain)
# out_dir = '../pre_trained_model/' + args.domain
U.print_args(args)

# Arguments:
# 2020-04-12 00:43:37,032 INFO   algorithm: adam
# 2020-04-12 00:43:37,032 INFO   aspect_size: 14
# 2020-04-12 00:43:37,032 INFO   batch_size: 32
# 2020-04-12 00:43:37,032 INFO   command: train.py --domain ty --emb-name ../preprocessed_data/ty/w2v_embedding --vocab-size 70000
# 2020-04-12 00:43:37,032 INFO   domain: ty
# 2020-04-12 00:43:37,032 INFO   emb_dim: 200
# 2020-04-12 00:43:37,032 INFO   emb_name: ../preprocessed_data/ty/w2v_embedding
# 2020-04-12 00:43:37,032 INFO   epochs: 15
# 2020-04-12 00:43:37,032 INFO   maxlen: 256
# 2020-04-12 00:43:37,032 INFO   neg_size: 5
# 2020-04-12 00:43:37,032 INFO   ortho_reg: 0.1
# 2020-04-12 00:43:37,032 INFO   out_dir_path: output
# 2020-04-12 00:43:37,032 INFO   seed: 1234
# 2020-04-12 00:43:37,032 INFO   vocab_size: 70000
# Reading data from  ty
#  Creating vocab ...
#  7864333 total words, 78215 unique words
#   keep the top 70000 words
#  Reading dataset ...
#   train set
#    <num> hit rate: 0.87%, <unk> hit rate: 0.11%
#   test set
#    <num> hit rate: 0.57%, <unk> hit rate: 4.02%
# Number of training examples:  963530
# Length of vocab:  70003

assert args.domain == 'ty'

###### Get test data #############
#
vocab, train_x, test_x, overall_maxlen = dataset.get_data(args.domain, vocab_size=args.vocab_size, maxlen=args.maxlen)
# pad the test data set.
test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)
test_length = test_x.shape[0]
splits = []
# put test_data into batches.
for i in range(1, test_length // args.batch_size):
    splits.append(args.batch_size * i)
if test_length % args.batch_size:
    splits += [(test_length // args.batch_size) * args.batch_size]
test_x = np.split(test_x, splits)

############# Build model architecture, same as the model used for training #########

## Load the save model parameters
model = load_model(out_dir + '/model_param',
                   custom_objects={"Attention": Attention, "Average": Average, "WeightedSum": WeightedSum,
                                   "MaxMargin": MaxMargin, "WeightedAspectEmb": WeightedAspectEmb,
                                   "max_margin_loss": U.max_margin_loss},
                   compile=True)

## Create a dictionary that map word index to word
# id2word.
vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w

test_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()],
                     [model.get_layer('att_weights').output, model.get_layer('p_t').output])
att_weights, aspect_probs = [], []
for batch in tqdm(test_x):
    cur_att_weights, cur_aspect_probs = test_fn([batch, 0])
    att_weights.append(cur_att_weights)
    aspect_probs.append(cur_aspect_probs)

att_weights = np.concatenate(att_weights)
aspect_probs = np.concatenate(aspect_probs)

######### Topic weight ###################################

topic_weight_out = open(out_dir + '/topic_weights', 'wt', encoding='utf-8')
labels_out = open(out_dir + '/labels.txt', 'wt', encoding='utf-8')
print('Saving topic weights on test sentences...')
for probs in aspect_probs:
    labels_out.write(str(np.argmax(probs)) + "\n")
    weights_for_sentence = ""
    for p in probs:
        weights_for_sentence += str(p) + "\t"
    weights_for_sentence.strip()
    topic_weight_out.write(weights_for_sentence + "\n")
print(aspect_probs)

## Save attention weights on test sentences into a file
att_out = open(out_dir + '/att_weights', 'wt', encoding='utf-8')
print('Saving attention weights on test sentences...')
test_x = np.concatenate(test_x)
for c in range(len(test_x)):
    att_out.write('----------------------------------------\n')
    att_out.write(str(c) + '\n')

    word_inds = [i for i in test_x[c] if i != 0]
    line_len = len(word_inds)
    weights = att_weights[c]
    weights = weights[(overall_maxlen - line_len):]

    words = [vocab_inv[i] for i in word_inds]
    att_out.write(' '.join(words) + '\n')
    for j in range(len(words)):
        att_out.write(words[j] + ' ' + str(round(weights[j], 3)) + '\n')
