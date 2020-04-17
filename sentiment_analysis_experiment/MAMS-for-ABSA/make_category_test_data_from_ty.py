import pandas as pd
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
import gensim
import regex
from data_process.utils import *
from ast import literal_eval
import pickle
from src.module.utils.constants import UNK, PAD_INDEX, ASPECT_INDEX
import csv

# bert-base-uncased model .
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

cd = {
    'food': 0,
    'service': 1,
    'staff': 2,
    'price': 3,
    'ambience': 4,
    'menu': 5,
    'place': 6,
    'renovation': 7
}


def process_sent(sent):
    sent = regex.sub('\S*@\S*\s?', '', sent)  # remove emails
    sent = regex.sub('\s+', ' ', sent)  # remove newline chars
    sent = regex.sub("\'", "", sent)  # remove single quotes
    # This lowercases, tokenizes, de-accents (optional). – the output are final tokens
    sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
    return ' '.join(sent)


categories = list(cd.keys())
test_file = '/home/yiyi/Documents/masterthesis/CPD/topic_modeling/test_categories.csv'
df = pd.read_csv(test_file)

df = df[df['categories'].astype(bool)]

data = [process_sent(x) for x in df['sents']]
categories_text = df['categories'].to_list()
data_categories = [(sent, category) for sent, category in zip(data, categories_text)]
print(data_categories[:10])

w2i = 'data/MAMS-ACSA/processed/word2index.pickle'
with open(w2i, 'rb') as file:
    word2index = pickle.load(file)

sentence = []
aspect = []
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
pieces = []
for d in data_categories:
    text, category = d
    if literal_eval(category) != []:
        for cat in literal_eval(category):
            # if cat == 'renovation':
            #     continue
            # else:
            pieces.append((text, cat))
            sentence.append(g(text))
            aspect.append(cd[cat])
            bert_sentence = bert_tokenizer.tokenize(text)
            bert_aspect = bert_tokenizer.tokenize(cat)
            bert_token.append(bert_tokenizer.convert_tokens_to_ids(
                ['[CLS]'] + bert_sentence + ['[SEP]'] + bert_aspect + ['[SEP]']))
            bert_segment.append([0] * (len(bert_sentence) + 2) + [1] * (len(bert_aspect) + 1))
            assert len(bert_token[-1]) == len(bert_segment[-1])
max_length = lambda x: max([len(y) for y in x])
sentence_max_len = max_length(sentence)
bert_max_len = max_length(bert_token)
num = len(pieces)
for i in range(num):
    sentence[i].extend([0] * (sentence_max_len - len(sentence[i])))
    bert_token[i].extend([0] * (bert_max_len - len(bert_token[i])))
    bert_segment[i].extend([0] * (bert_max_len - len(bert_segment[i])))
sentence = np.asarray(sentence, dtype=np.int32)
aspect = np.asarray(aspect, dtype=np.int32)
bert_token = np.asarray(bert_token, dtype=np.int32)
bert_segment = np.asarray(bert_segment, dtype=np.int32)
print(sentence[:1])
print(sentence[:1])
print(bert_token[:1])
print(bert_segment[:1])

path = 'data/MAMS-ACSA/processed/test_ty.npz'
np.savez(path, sentence=sentence, aspect=aspect, bert_token=bert_token, bert_segment=bert_segment)

pathcsv = 'data/MAMS-ACSA/processed/test_ty_text.csv'
with open(pathcsv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['text', 'category'])
    for piece in pieces:
        text, cat = piece
        writer.writerow([text, cat])
