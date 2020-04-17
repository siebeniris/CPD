from nltk.tokenize import sent_tokenize, word_tokenize
from more_itertools import consecutive_groups


def load_sentiment_terms(file):
    # pos, neg
    # adjectives = ['JJ','JJR','JJS']
    terms = []
    with open(file) as input:
        for line in input:
            if not line.startswith(';'):
                terms.append(line.strip())
    return list(set(terms))

def get_phrases():
    import itertools
    sentence = 'A stunning weekend getaway at Phinda Mountain Lodge .'
    offsets = [(11,18), (19, 26), (30, 36), (37, 45), (46,51)]

    offset = 0
    offset_dict = dict()
    tokens = word_tokenize(sentence)
    print(tokens)
    for idx, word in enumerate(tokens):
        offset_dict[(offset, offset+len(word))] = idx
        offset += len(word)+1

    print(offset_dict)
    indices = [offset_dict[x] for x in offsets]
    for group in consecutive_groups(indices):
        index=list(group)
        start, end = index[0], index[-1]
        print(tokens[start:end])


if __name__ == '__main__':
    get_phrases()