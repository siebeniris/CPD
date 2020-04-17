import os
from xml.etree.ElementTree import parse

import spacy
import regex
import numpy as np

from pytorch_pretrained_bert import BertTokenizer
from data_process.utils import *
from src.module.utils.constants import UNK, PAD_INDEX, ASPECT_INDEX

url = regex.compile('(<url>.*</url>)')
spacy_en = spacy.load('en')


def check(x):
    return len(x) >= 1 and not x.isspace()


def tokenizer(text):
    tokens = [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]
    return list(filter(check, tokens))


# bert-base-uncased model .
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# aspect term data.
base_path = './data/MAMS-ATSA'

raw_train_path = os.path.join(base_path, 'raw/train.xml')
raw_val_path = os.path.join(base_path, 'raw/val.xml')
raw_test_path = os.path.join(base_path, 'raw/test.xml')

remove_list = ['conflict']
train_data = parse_sentence_term(raw_train_path, True)
train_data = category_filter(train_data, remove_list)
word2index, index2word = build_vocab(train_data, max_size=None, min_freq=0)


# parse sentence term.
def parse_sentence_term_ty(path, lowercase=False):
    tree = parse(path)
    root = tree.getroot()
    data = []
    split_char = '__split__'
    for sentence in root.iter('sentence'):
        text = sentence.find('text')
        if text is None:
            continue
        text = text.text
        if lowercase:
            text = text.lower()
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:
            continue
        for aspectTerm in aspectTerms:
            term = aspectTerm.get('term')
            if lowercase:
                term = term.lower()
            start = aspectTerm.get('from')
            end = aspectTerm.get('to')
            piece = text + split_char + term + split_char + start + split_char + end
            data.append(piece)
    return data


## save term data
def save_term_test_data_ty(data, word2index, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    sentence = []
    aspect = []
    context = []
    bert_token = []
    bert_segment = []
    td_left = []
    td_right = []
    f = lambda x: word2index[x] if x in word2index else word2index[UNK]
    g = lambda x: list(map(f, tokenizer(x)))

    for piece in data:
        text, term, start, end = piece.split('__split__')
        start, end = int(start), int(end)
        assert text[start: end] == term
        sentence.append(g(text))
        aspect.append(g(term))

        left_part = g(text[:start])
        right_part = g(text[end:])

        context.append(left_part + [ASPECT_INDEX] + right_part)
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
    num = len(data)
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
    context = np.asarray(context, dtype=np.int32)
    bert_token = np.asarray(bert_token, dtype=np.int32)
    bert_segment = np.asarray(bert_segment, dtype=np.int32)
    td_left = np.asarray(td_left, dtype=np.int32)
    td_right = np.asarray(td_right, dtype=np.int32)

    np.savez(path, sentence=sentence, aspect=aspect, context=context, bert_token=bert_token, bert_segment=bert_segment,
             td_left=td_left, td_right=td_right)


if __name__ == '__main__':
    input_dir = '/home/yiyi/Documents/masterthesis/CPD/data/aspect_extraction/output/'
    filename = '0a5c0a4c-36f7-46c4-9f13-91f52ba45ea5'
    input_file = os.path.join(input_dir, filename + '.xml')
    data = parse_sentence_term_ty(input_file, True)
    print(len(data))
    output_dir = '/home/yiyi/Documents/masterthesis/CPD/data/ABSA/processed'
    output_file = os.path.join(output_dir, filename + '.npz')
    save_term_test_data_ty(data, word2index, output_file)
