#  python infer.py --domain ty --vocab-size 78215
import os
from xml.etree import ElementTree
from ast import literal_eval

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
    # for each SubElement Sentence.
    sentence = root.find(path_)

    orig_text = sentence.attrib.get('orig')
    lemmas = sentence.text.split()

    indices_sentence = literal_eval(sentence.attrib.get('indices'))
    nounIndices = literal_eval(sentence.attrib.get('nounIndices'))
    nounPhrases = literal_eval(sentence.attrib.get('nounPhrases'))

    # the words
    words = [vocab_inv[i] for i in word_inds]
    att_out.write(' '.join(words) + '\n')

    # ordered by weight.
    ind_word_weight =[]

    for elem in sorted_weights_dict:
        ind, weight = elem
        word_indices = indices_sentence[ind]
        word = lemmas[ind]
        # only choose the aspects with weight more than 0.07 and not in sentiment terms.
        if word not in sentiments and weight >= 0.07:
            ind_word_weight.append((word_indices, word, weight))

        att_out.write(word + ' ' + str(weight) + '\n')


    ind_word_weight_sorted = sorted(ind_word_weight, key=lambda x:x[0])

    # post process the aspect terms.
    # aspectTerms_phrases =[]
    # for offsets_lemma, aspect_term, weight in ind_word_weight:
    #     # check if the aspect_term is a noun phrase, if it is, get the indices.
    #     for idx, noun_phrase in enumerate(nounPhrases):
    #         # if aspect_term in noun_phrase:
    #         offsets_noun_phrase = nounIndices[idx]
    #         if offsets_lemma[0] >= offsets_noun_phrase[0] and offsets_lemma[1] <= offsets_noun_phrase[1]:
    #             aspectTerms_phrases.append((offsets_noun_phrase, noun_phrase))
    #     if not any([offsets_lemma[0] >= offsets_noun_phrase[0] and offsets_lemma[1] <= offsets_noun_phrase[1]
    #                 for offsets_noun_phrase, _ in list(set(aspectTerms_phrases))]):
    #         aspectTerms_phrases.append((offsets_lemma, aspect_term))
    #
    # sorted_aspects = sorted(list(set(aspectTerms_phrases)), key=lambda x: x[0])
    # processed_aspects = []
    # for offsets, aspect_term in sorted_aspects:
    #     orig_term = orig_text[offsets[0]:offsets[1]]
    #     starting = aspect_term.split()[0]
    #     if starting in sentiments:
    #         new_starting = offsets[0] + len(starting) + 1
    #         new_term = orig_text[new_starting: offsets[1]]
    #         processed_aspects.append(((new_starting, offsets[1]), new_term))
    #     else:
    #         processed_aspects.append((offsets, orig_term))

    sentence.text = None
    text = ElementTree.SubElement(sentence, 'text')
    text.text = orig_text
    group = ElementTree.SubElement(sentence, 'aspectTerms')

    for offsets, aspect_term, _ in ind_word_weight_sorted:
        aspectTerm = ElementTree.SubElement(group, 'aspectTerm', {
            'from': str(offsets[0]),
            'term': aspect_term,
            'to': str(offsets[1])
        })

    print('sentence:{}'.format(c), end='\r')

    # delete unnecessary attributes in sentence subelement.
    sentence.attrib.pop('indices', None)
    sentence.attrib.pop('nounIndices', None)
    sentence.attrib.pop('nounPhrases', None)
    sentence.attrib.pop('orig', None)
    tree.write('output.xml')

timer.stop()