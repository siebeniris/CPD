# https://github.com/MilaNLProc/contextualized-topic-models/blob/master/contextualized_topic_models/utils/data_preparation.py
# data preparation for contextualized-topic-models.
import csv
import string
import pandas as pd


import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed

from utils.preprocess import parseSentence
from utils.timer import Timer

def get_bag_of_words(data, min_length):

    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]
    return np.array(vect)


def bert_embeddings_from_data(text_data, sbert_model_to_load):
    """
    GEt bert embeddings for data
    :param text_data: a list of sentences/
    :param sbert_model_to_load:
    :return:
    """
    model = SentenceTransformer(sbert_model_to_load)

    train_text = list(map(lambda x: x, text_data))

    return np.array(model.encode(train_text))


def bert_embeddings_from_list(texts, sbert_model_to_load):
    model = SentenceTransformer(sbert_model_to_load)
    return np.array(model.encode(texts))


class TextHandler:

    def __init__(self, filepath):
        self.filepath = filepath
        self.vocab_dict = {}
        self.vocab = []
        self.index_dd = None
        self.idx2token = None
        self.training_bow = None

    def load_textdata(self):
        """
        Loads a text file
        :param text_file:
        :return:
        """
        timer = Timer()
        timer.start()

        df = pd.read_csv(self.filepath)
        df = df.replace(np.nan, '', regex=True)
        df.text = df.text.astype(str)
        df.title = df.title.astype(str)

        # pandas dataframe. words = title+text.
        df["words"] = df.title + ' .' + df.text
        # filter the english content.
        en_df = df[df.lang == 'en']
        en_df = en_df[en_df.words.notnull()]

        docs = en_df.words.to_list()
        # sentence tokenize inside parseSentence.
        processed = Parallel(n_jobs=-1)(delayed(parseSentence)(line) for line in docs)
        data = [item for sublist in processed for item in sublist]
        timer.stop()

        return data

    def prepare(self):

        self.data = self.load_textdata()
        concatenate_text = ""
        for line in self.data:
            print(line)
            line = line.strip()
            concatenate_text += line + " "
        concatenate_text = concatenate_text.strip()

        self.vocab = list(set(concatenate_text.split()))

        for index, vocab in list(zip(range(0, len(self.vocab)), self.vocab)):
            self.vocab_dict[vocab] = index

        self.index_dd = np.array(list(map(lambda y: np.array(list(map(lambda x:
                                                                      self.vocab_dict[x], y.split()))), self.data)))
        self.idx2token = {v: k for (k, v) in self.vocab_dict.items()}
        self.bow = get_bag_of_words(self.index_dd, len(self.vocab))



