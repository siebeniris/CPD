#  python infer.py --domain ty --vocab-size 78215
import os
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from ast import literal_eval
from nltk.corpus import wordnet as wn
from preprocess import load_sentiment_terms


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
att_weights =  []
for batch in tqdm(test_x):
    cur_att_weights, _ = test_fn([batch, 0])
    att_weights.append(cur_att_weights)

att_weights = np.concatenate(att_weights)

# get all nouns from wordnet
# nouns = {x.name().split('.',1)[0] for x in wn.all_synsets('n')}

with open(test_xml, 'rt') as file:
    tree = ElementTree.parse(file)
    root = tree.getroot()


## Save attention weights on test sentences into a file
sentiments = load_sentiment_terms('sentiments.txt')
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
    weights_dict = {idx: round(weight,3) for idx, weight in enumerate(weights)}
    # how many weights to choose:
    weight_nr = line_len//3+2
    import operator
    # sort the weights by descending order
    sorted_weights_dict =sorted(weights_dict.items(), key= operator.itemgetter(1), reverse= True)
    # [(word_ind, weight)]

    path_ = './/sentence[@id="{}"]'.format(str(c))
    sentence = root.find(path_)
    orig_text=sentence.attrib.get('orig')

    indices_sentence = literal_eval(sentence.attrib.get('indices'))

    noun_phrases = literal_eval(sentence.attrib.get('nounIndices'))
    # the words
    words = [vocab_inv[i] for i in word_inds]
    att_out.write(' '.join(words) + '\n')
    count = 0
    indices = []
    # ordered by weight.
    ind_word_weight =[]
    for elem in sorted_weights_dict:
        ind, weight = elem
        word = words[ind]
        if count < weight_nr :

            if word not in sentiments:
                ind_word_weight.append((ind, word, weight))
                att_out.write(word + ' '+ str(weight)+'\n')
                count +=1

        # if count < weight_nr:
        #     # word not pos/neg.
        #     if word not in sentiments:
        #         indices.append(ind)
        #         att_out.write(word + ' ' + str(weight) + '\n')
        #         count += 1

    # aspects_indices = sorted([indices_sentence[x] for x,_,_ in indices])

    # aspect_nouns_indiecs =[]
    # for (x,y) in aspects_indices:
    #     for (z,s) in noun_phrases:
    #         if x >= z and y <= s :
    #             aspect_nouns_indiecs.append((z,s))
    #         if y<z and x <z:
    #             aspect_nouns_indiecs.append((x,y))
    #         if x>s and y>s:
    #             aspect_nouns_indiecs.append((x,y))
    # aspects_nouns = [orig_text[x:y] for (x,y) in list(set(aspects_indices))]
    # aspect_nouns = [orig_text[x:y] for (x,y) in list(set(aspect_nouns_indiecs))]
    ind_word_weight_sorted= sorted_weights_dict =sorted(ind_word_weight, key=lambda x:x[0])
    group = ElementTree.SubElement(sentence, 'aspectTerms',{
        'ind_word_weight': str(ind_word_weight_sorted)
        # 'terms': str(aspects_nouns),
        # 'indices': str(list(set(aspects_indices))),
        # "processed_terms": str(aspect_nouns),
        # "processed_indices": str(list(set(aspect_nouns_indiecs)))
    })


    tree.write('output.xml')

