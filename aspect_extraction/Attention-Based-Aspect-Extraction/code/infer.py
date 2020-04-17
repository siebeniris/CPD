#  python infer.py --domain ty --vocab-size 78215
import os
from xml.etree import ElementTree
from ast import literal_eval
from more_itertools import consecutive_groups
import string

import numpy as np
from keras.models import load_model
from tqdm import tqdm
import keras.backend as K
from keras.preprocessing import sequence

import utils as U
import reader as dataset
from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin
from postprocess import load_sentiment_terms
from timer import Timer

######### Get hyper-params in order to rebuild the model architecture ###########
# The hyper parameters should be exactly the same as those used for training

parser = U.add_common_args()
args = parser.parse_args()

out_dir = os.path.join(args.out_dir_path , args.domain)
U.print_args(args)


assert args.domain == 'ty'

###### Get test data #############
test_filename = '0a5c0a4c-36f7-46c4-9f13-91f52ba45ea5'
input_dir = '/home/yiyi/Documents/masterthesis/CPD/data/aspect_extraction'
test_xml = os.path.join(input_dir, test_filename + '.xml')
output_path = os.path.join(input_dir, 'output', test_filename+'.xml')


# get data.
vocab, train_x, test_x, overall_maxlen = dataset.get_data_ty(args.domain, test_xml, vocab_size=args.vocab_size, maxlen=args.maxlen)
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
att_weights = []
for batch in tqdm(test_x):
    cur_att_weights, _ = test_fn([batch, 0])
    att_weights.append(cur_att_weights)

att_weights = np.concatenate(att_weights)

# get all nouns from wordnet
# nouns = {x.name().split('.',1)[0] for x in wn.all_synsets('n')}

timer = Timer()
timer.start()
with open(test_xml, 'rt') as file:
    tree = ElementTree.parse(file)
    root = tree.getroot()


## Save attention weights on test sentences into a file
sentiments = load_sentiment_terms('sentiments_all.txt')

att_out = open(out_dir + '/att_weights', 'wt', encoding='utf-8')
print('Saving attention weights on test sentences...')
test_x = np.concatenate(test_x)
# id of the sentences.
for c in range(len(test_x)):

    att_out.write('----------------------------------------\n')
    att_out.write(str(c) + '\n')

    word_inds = [i for i in test_x[c] if i != 0]
    line_len = len(word_inds)
    weights = att_weights[c]
    weights = weights[(overall_maxlen - line_len):]
    # word_ind, weight
    weights_dict = {idx: round(weight, 3) for idx, weight in enumerate(weights)}

    sorted_weights_dict = sorted(weights_dict.items(), key= lambda x: x[1], reverse=True)
    # how many weights to choose: every three word, choose one aspect.
    weight_nr = line_len//3+2

    path_ = './/sentence[@id="{}"]'.format(str(c))
    # for each SubElement Sentence.
    sentence = root.find(path_)

    indices_lemmas = literal_eval(sentence.attrib.get('indices'))
    # offset: idx
    ind2offset = literal_eval(sentence.attrib.get('offsetDict'))


    orig_text = sentence.attrib.get('orig')
    lemmas = sentence.text.split()

    # the write to attr_weight.
    words = [vocab_inv[i] for i in word_inds]
    att_out.write(' '.join(words) + '\n')
    att_out.write(sentence.text + '\n')

    # ordered by weight.
    aspect_indices = []  # index 2 word in original text.
    count=0
    for elem in sorted_weights_dict:
        ind, weight = elem
        # the index of word in the original text.
        word_idx = indices_lemmas[ind]  # 2
        word_start, word_end = ind2offset[word_idx]  # (11,18)
        word = lemmas[ind]  # get the lemma word

        # only choose the aspects with weight more than 0.035 and not in sentiment terms and not
        # a punctuation mark.
        if count < weight_nr:
            if word not in sentiments and word not in string.punctuation and weight >= 0.035:
                aspect_indices.append(word_idx)
                count += 1

        att_out.write(word + ' ' + str(weight) + '\n')

    aspects_terms = []

    for group in consecutive_groups(sorted(aspect_indices)):
        index = list(group)
        offset0, offset1 = ind2offset[index[0]], ind2offset[index[-1]]
        start, end = offset0[0], offset1[-1]
        aspect_term = orig_text[start:end]
        aspects_terms.append(((start, end), aspect_term))


    sentence.text = None
    text = ElementTree.SubElement(sentence, 'text')
    text.text = orig_text
    group = ElementTree.SubElement(sentence, 'aspectTerms')

    for (start, end), aspect_term in aspects_terms:
        aspectTerm = ElementTree.SubElement(group, 'aspectTerm', {
            'from': str(start),
            'term': aspect_term,
            'to': str(end)
        })

    print('sentence:{}'.format(c), end='\r')

    # delete unnecessary attributes in sentence subelement.
    sentence.attrib.pop('orig', None)
    sentence.attrib.pop('offsetDict', None)
    sentence.attrib.pop('indices', None)
    sentence.attrib.pop('nounIndices', None)
    sentence.attrib.pop('nounPhrases', None)
    tree.write(output_path)

timer.stop()