import gensim
import codecs


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def main(domain):
    source = '../preprocessed_data/%s/train.txt' % (domain)
    model_file = '../preprocessed_data/%s/w2v_embedding' % (domain)
    sentences = MySentences(source)
    # https://radimrehurek.com/gensim/models/word2vec.html
    model = gensim.models.Word2Vec(sentences, size=200, window=10, min_count=2, workers=4)
    model.save(model_file)


print('Pre-training word embeddings ...')
main('ty')
