import os
from data_process.utils import *
from src.module.utils.constants import UNK, PAD_INDEX, ASPECT_INDEX

from pytorch_pretrained_bert import BertTokenizer
import numpy as np
########################################################333
base_path = './data/MAMS-ATSA'
raw_train_path = os.path.join(base_path, 'raw/train.xml')
raw_val_path = os.path.join(base_path, 'raw/val.xml')
raw_test_path = os.path.join(base_path, 'raw/test.xml')
lowercase = True

# parse the sentence from the xml file.
train_data = parse_sentence_term(raw_train_path, lowercase=lowercase)
val_data = parse_sentence_term(raw_val_path, lowercase=lowercase)
test_data = parse_sentence_term(raw_test_path, lowercase=lowercase)

print('train_data parsed ==>')
print(train_data[:1])

train_data_filtered = category_filter(train_data, ['conflict'])
remove_list = ['conflict']
print('train data after category_filter=>')
print(train_data_filtered[:1])
##### test data
word2index, index2word = build_vocab(train_data, max_size=None, min_freq=0)

print('vocab_size=', len(word2index))


### the data used for training/test
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = []
aspect = []
label = []
bert_token = []
bert_segment = []
f = lambda x: word2index[x] if x in word2index else word2index[UNK]
g = lambda x: list(map(f, tokenizer(x)))
d = {
    'positive': 0,
    'negative': 1,
    'neutral': 2,
    'conflict': 3
}
cd = {
    'food': 0,
    'service': 1,
    'staff': 2,
    'price': 3,
    'ambience': 4,
    'menu': 5,
    'place': 6,
    'miscellaneous': 7
}

# check train_data/
sentence = []
aspect = []
label = []
context = []
bert_token = []
bert_segment = []
td_left = []
td_right = []
f = lambda x: word2index[x] if x in word2index else word2index[UNK]
g = lambda x: list(map(f, tokenizer(x)))
d = {
    'positive': 0,
    'negative': 1,
    'neutral': 2,
    'conflict': 3
}



for piece in train_data:
    text, term, polarity, start, end = piece.split('__split__')
    start, end = int(start), int(end)
    assert text[start: end] == term
    # need term
    sentence.append(g(text))
    aspect.append(g(term))
    label.append(d[polarity])
    left_part = g(text[:start])
    right_part = g(text[end:])
    context.append(left_part + [ASPECT_INDEX] + right_part)

    # using text and term to calculate bert_token and bert_segment
    bert_sentence = bert_tokenizer.tokenize(text)
    bert_aspect = bert_tokenizer.tokenize(term)
    bert_token.append(
        bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_sentence + ['[SEP]'] + bert_aspect + ['[SEP]']))
    bert_segment.append([0] * (len(bert_sentence) + 2) + [1] * (len(bert_aspect) + 1))
    td_left.append(g(text[:end]))
    td_right.append(g(text[start:])[::-1])
    assert len(bert_token[-1]) == len(bert_segment[-1])
max_length = lambda x: max([len(y) for y in x])
sentence_max_len = max_length(sentence)
aspect_max_len = max_length(aspect)
context_max_len = max_length(context)
bert_max_len = max_length(bert_token)
td_left_max_len = max_length(td_left)
td_right_max_len = max_length(td_right)
num = len(train_data)
for i in range(num):
    sentence[i].extend([0] * (sentence_max_len - len(sentence[i])))
    aspect[i].extend([0] * (aspect_max_len - len(aspect[i])))
    context[i].extend([0] * (context_max_len - len(context[i])))
    bert_token[i].extend([0] * (bert_max_len - len(bert_token[i])))
    bert_segment[i].extend([0] * (bert_max_len - len(bert_segment[i])))
    td_left[i].extend([0] * (td_left_max_len - len(td_left[i])))
    td_right[i].extend([0] * (td_right_max_len - len(td_right[i])))
sentence = np.asarray(sentence, dtype=np.int32)
aspect = np.asarray(aspect, dtype=np.int32)
label = np.asarray(label, dtype=np.int32)
context = np.asarray(context, dtype=np.int32)
bert_token = np.asarray(bert_token, dtype=np.int32)
bert_segment = np.asarray(bert_segment, dtype=np.int32)
td_left = np.asarray(td_left, dtype=np.int32)
td_right = np.asarray(td_right, dtype=np.int32)

print(sentence.shape)
print(aspect.shape)
print(label.shape)
print(bert_token.shape)
print(bert_segment.shape)