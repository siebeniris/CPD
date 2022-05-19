import rootpath
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import json

from gensim.models import Word2Vec
from spellchecker import SpellChecker

from utils.timer import Timer

# use the distance 1 -> fast.
spell_checker = SpellChecker(distance=1, language='en')

# SpellChecker cannot be made public 

def init_model():
    spell = SpellChecker()
    filename = '422#793f568b-3393-4e81-a940-6b2a84ca44bb'

    filepath = os.path.join(data_dir, filename)
    print('loading file:', filepath)
    df = pd.read_csv(filepath)
    df['date'] = df['date'].astype('datetime64')
    df = df[df['date'] >= np.datetime64("2015-01-01")]
    df['lemma'] = df['lemma'].astype(str)
    lemmas = [lemma.lower().split() for lemma in df['lemma']]
    corrected_lemmas = []
    for x in tqdm(lemmas):
        corrected_lemmas.append([spell.correction(lemma) for lemma in x])
    model = Word2Vec(corrected_lemmas, size=200, window=6, min_count=2, workers=8, iter=200, negative=5)
    return model


def train_model(input_dir, output_dir):
    """

    :param input_dir:
    :param output_dir:
    :return:
    """
    count = 0
    for filename in os.listdir(input_dir):
        output_path = os.path.join(output_dir, filename)
        if not os.path.exists(output_path):

            timer = Timer()
            timer.start()

            filepath = os.path.join(input_dir, filename)
            print('loading file:', filepath)
            df = pd.read_csv(filepath)

            df['date'] = df['date'].astype('datetime64')
            df = df[df['date'] >= np.datetime64("2015-01-01")]
            df['lemma'] = df['lemma'].astype(str)
            lemmas = [lemma.lower().split() for lemma in df['lemma']]
            # lemmas_corrected= Parallel(n_jobs=-1)(delayed(spell_check)(x) for x in lemmas)
            # lemmas_corrected = [[spell.correction(lemma) for lemma in x] for x in lemmas]
            lemmas_corrected = []
            for x in tqdm(lemmas):
                lemmas_corrected.append(json.dumps(apply_spellchecker(spell_checker, x)))

            df['lemma'] = lemmas_corrected
            df.to_csv(output_path)

            print('save to ', output_path)
            # model.build_vocab(lemmas_corrected, update=True)
            # model.train(lemmas_corrected, total_examples=len(lemmas_corrected), epochs=2)
            timer.stop()
            count += 1


if __name__ == '__main__':
    root = rootpath.detect()
    data_dir = os.path.join(root, 'data/sentiment_analysis/results')
    spellchecked_dir = os.path.join(root, 'data/spellchecked')
    timer = Timer()
    timer.start()
    # model = init_model()
    # model = Word2Vec.load("word2vec_init.model")
    timer.stop()
    train_model(data_dir, spellchecked_dir)
    # model.save("word2vec_lemmas.model")
