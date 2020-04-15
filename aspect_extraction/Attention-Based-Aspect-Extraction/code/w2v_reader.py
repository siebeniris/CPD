import os
import logging
import numpy as np
from sklearn.cluster import KMeans
import pymorphy2

import gensim.downloader as api

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class W2VReader:
    def __init__(self):

        self.embeddings={}
        emb_matrix=[]

        model = api.load('word2vec-google-news-300')

        self.emb_dim = model.vector_size
        for word in model.vocab:
            self.embeddings[word] = list(model[word])
            emb_matrix.append(list(model[word]))

        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)
        self.aspect_size = None
        logger.info('  #vectors: %i, #dimensions: %i' % (self.vector_size, self.emb_dim))

    def get_emb_given_word(self, word):
        try:
            return self.embeddings[word]
        except KeyError:
            return None

    def get_emb_matrix_given_vocab(self, vocab, emb_matrix):
        counter = 0.
        for word, index in vocab.items():
            try:
                emb_matrix[index] = self.embeddings[word]
                counter += 1
            except KeyError:
                pass

        logger.info(
            '%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100 * counter / len(vocab)))
        # L2 normalization
        norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
        return norm_emb_matrix

    def get_aspect_matrix(self, n_clusters=0):
        self.aspect_size = n_clusters
        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb_matrix)
        km_aspects = km.cluster_centers_
        aspects = km_aspects
        # L2 normalization
        norm_aspect_matrix = aspects / np.linalg.norm(aspects, axis=-1, keepdims=True)
        return norm_aspect_matrix

    def get_emb_dim(self):
        return self.emb_dim