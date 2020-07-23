import numpy as np
import os
from random import shuffle
import re
import zipfile
import lxml.etree
import urllib
from collections import Counter
import nltk
import enchant
import io
import itertools
import collections
from gensim.models import FastText
import warnings
import rootpath

root_dir = rootpath.detect()
fasttext_model_path = os.path.join(root_dir, 'embeddings', "model.bin")
fasttext_model = FastText.load(fasttext_model_path)
enchant_us = enchant.Dict('en_US')


fasttext_min_similarity = 0.6

# function to identify possible misspellings
def include_spell_mistake(word, similar_word, score):
    edit_distance_threshold = 1 if len(word) <= 4 else 2
    score_1 = score > fasttext_min_similarity
    score_2 = len(similar_word) > 3
    score_3 = not enchant_us.check(similar_word)
    score_4 = word[0] == similar_word[0]
    score_5 = nltk.edit_distance(word, similar_word) <= edit_distance_threshold
    score = score_1 + score_2 + score_3 + score_4 + score_5
    if score > 3:
        return True
    else:
        return False


def spell_checker(word):
    w2m = []
    most_similar = fasttext_model.wv.most_similar(word, topn=50)
    for similar_word, score in most_similar:
        if include_spell_mistake(word, similar_word, score):
            w2m.append(similar_word)
    output = {'prediction': w2m[:3]}
    return output


if __name__ == '__main__':
    word = "comfortable"
    checked = spell_checker(word)
    print(checked)