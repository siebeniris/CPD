import regex
import string
import os
from functools import partial

import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.util import minibatch
from nltk.corpus import wordnet
from joblib import Parallel, delayed
import rootpath

from utils.timer import Timer
from utils.contractions import contractions_dict


#  https://github.com/explosion/spaCy/issues/2627
def prevent_sentence_boundaries(doc):
    for token in doc:
        if not can_be_sentence_start(token):
            token.is_sent_start = False
    return doc


def can_be_sentence_start(token):
    if token.i == 0:
        return True
    # We're not checking for is_title here to ignore arbitrary titlecased
    # tokens within sentences
    # elif token.is_title:
    #    return True
    elif token.nbor(-1).is_punct:
        return True
    elif token.nbor(-1).is_space:
        return True
    else:
        return False


def remove_repeated_characters(word):
    pattern = regex.compile(r"(\w*)(\w)\2(\w*)")
    substitution_pattern = r"\1\2\3"
    while True:
        if wordnet.synsets(word):
            return word
        new_word = pattern.sub(substitution_pattern, word)
        if new_word != word:
            word = new_word
            continue
        else:
            return new_word


def expand_contractions(text, contractions_dict):
    """
    Expand the contractions regarding the contractions_dict.
    :param text:
    :param contractions_dict:
    :return:
    """
    contractions_pattern = regex.compile('({})'.format('|'.join(contractions_dict.keys())),
                                         flags=regex.IGNORECASE | regex.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions_dict.get(match) \
            if contractions_dict.get(match) \
            else contractions_dict.get(match.lower())
        expanded_contraction = expanded_contraction
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = regex.sub("'", "", expanded_text)
    return expanded_text


def process_one_review(nlp, reviews):
    """
    Process one review, expand contractions, split into sentences. Then remove repeated characters,
     remove stop words, extract lemmas, remove punctuations.
    :param review:
    :return:
    """
    processed_sentences = []
    lemmas = []
    for review in reviews:
        try:
            review = expand_contractions(review, contractions_dict)
        except Exception:
            review = review
        doc = nlp(review)

        for sent in doc.sents:
            processed_sentence = []
            lemma = []
            for token in sent:
                # remove repeated characters for original sentence
                token_text = remove_repeated_characters(token.text)
                # strip the punctuations for each word
                punct_stripped = token_text.translate(str.maketrans('', '', string.punctuation)).strip()
                if len(punct_stripped) > 0:
                    processed_sentence.append(punct_stripped)
                if token_text not in STOP_WORDS:
                    # get the lemmas for later topic modeling.
                    if token.pos_ in ['PROPN', 'NOUN', 'VERB', 'ADJ'] and not token.is_punct:
                        # remove the punctuations in each word
                        t = token.lemma_.translate(str.maketrans('', '', string.punctuation)).strip()
                        if len(t) > 1:
                            lemma.append(t)

            processed_sentences.append(processed_sentence)
            lemmas.append(lemma)

    return list(zip(processed_sentences, lemmas))


def get_data_list(filepath):
    """
    Get the data list from each csv file.
    :param filepath:
    :return: processed texts/lemmas , dates, scores, and uids.
    """
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe(prevent_sentence_boundaries, before="parser")

    timer = Timer()
    timer.start()

    df = pd.read_csv(filepath)
    df = df.replace(np.nan, '', regex=True)
    df.text = df.text.astype(str)
    df.title = df.title.astype(str)

    sentences = []
    # pandas dataframe. words = title+text.
    for title, text in zip(df.title, df.text):
        if len(title) > 3 and len(text) > 3:
            sentences.append((title + '. ' + text))
        elif len(title) < 3 and len(text) > 3:
            sentences.append(text)
        else:
            sentences.append(np.nan)

    df['words'] = sentences
    # filter the english content.

    en_df = df[df.lang == 'en']
    en_df = en_df[en_df.words.notnull()]

    docs = en_df.words.to_list()

    # https://github.com/explosion/spaCy/blob/master/examples/pipeline/multi_processing.py
    # multiprocessing on the reviews.
    partitions = minibatch(docs, size=1000)

    executor = Parallel(n_jobs=-1, backend='multiprocessing', prefer='processes')
    do = delayed(partial(process_one_review, nlp))
    tasks = (do(batch) for batch in partitions)
    processed = executor(tasks)

    date = en_df['date'].to_list()
    score = en_df['score'].to_list()
    uid = en_df['uid'].to_list()

    timer.stop()

    return processed, date, score, uid


def get_data(filepath):
    """
    Unfold the data into sentenes, lemmas, uids.

    :param filepath: the path to the csv file
    :return:
    """

    processed, date, score, uid = get_data_list(filepath)
    lemmas, texts = [], []
    for comb in processed:
        if comb:
            text, lemma = zip(*comb)
            lemmas.append(lemma)
            texts.append(text)
        else:
            lemmas.append([])
            texts.append([])

    lemma_sents = [[' '.join(tokens) for tokens in sublist] for sublist in lemmas]
    sents = [[' '.join(tokens) for tokens in sublist] for sublist in texts]

    # flatten the lists
    dates = [[date[i] for _ in range(len(lemma_sents[i]))] for i in range(len(lemma_sents))]
    scores = [[score[i] for _ in range(len(lemma_sents[i]))] for i in range(len(lemma_sents))]
    uids = [[uid[i] for _ in range(len(lemma_sents[i]))] for i in range(len(lemma_sents))]

    lemma_sentss = [sent for sublist in lemma_sents for sent in sublist]
    sentss = [sent for sublist in sents for sent in sublist]
    datess = [sent for sublist in dates for sent in sublist]
    scoress = [sent for sublist in scores for sent in sublist]
    uidss = [sent for sublist in uids for sent in sublist]

    assert len(lemma_sentss) == len(sentss) == len(datess) == len(scoress) == len(uidss)

    return uidss, lemma_sentss, sentss, datess, scoress


if __name__ == '__main__':
    # test
    filepath = 'data/cleand_query_output_csv/0#f116f785-8626-48f3-a390-c0c4a03b5bd6'
    rootdir = rootpath.detect()
    fullpath = os.path.join(rootdir, filepath)
    timer = Timer()
    timer.start()
    uids, lemmas, sents, dates, scores = get_data(fullpath)
    print(uids[:10])
    print(lemmas[:10])
    print(sents[:10])
    print(dates[:10])
    timer.stop()
