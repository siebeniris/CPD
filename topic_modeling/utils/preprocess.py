import regex
import string
from nltk.corpus import wordnet

import numpy as np
import pandas as pd
from utils.contractions import contractions_dict
from joblib import Parallel, delayed


from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from utils.timer import Timer


def remove_repeated_characters(word):
    pattern = regex.compile(r"(\w*)(\w)\2(\w*)")
    substitution_pattern = r"\1\2\3"
    while True:
        if wordnet.synsets(word):
            return word
        new_word = pattern.sub(substitution_pattern,word)
        if new_word != word:
            word = new_word
            continue
        else:
            return new_word


def expand_contractions(text, contractions_dict):
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

def parseSentence(doc):
    lines = sent_tokenize(doc)
    # lowercased
    processed_doc=[]
    for line in lines:
        line = line.lower()
        # delete stop words and lemmatize.
        lmtzr = WordNetLemmatizer()

        stop = stopwords.words('english')
        # text_token = CountVectorizer().build_tokenizer()(line.lower())
        # expand the words with contractiosn
        try:
            expanded = expand_contractions(line, contractions_dict)
            # remove repeated characters for each word
            text_processed = [remove_repeated_characters(word) for word in word_tokenize(expanded)]
        except Exception:
            text_processed = [remove_repeated_characters(word) for word in word_tokenize(line)]

        # lemmatize words, remove stop words and digits.
        lemmas = [lmtzr.lemmatize(word) for word in text_processed if word not in stop if word.isalpha()]
        # remove all the punctuations.
        processed = [s.translate(str.maketrans('', '', string.punctuation)).strip() for s in lemmas]
        if len(processed) > 1:
            processed_doc.append(processed)
    return processed_doc


def get_data_list(filepath):
    """

    :param filepath:
    :return:
    """
    timer = Timer()
    timer.start()

    df = pd.read_csv(filepath)
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

