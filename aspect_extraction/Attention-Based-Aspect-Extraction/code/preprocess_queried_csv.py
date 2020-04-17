import pandas as pd
import os
import numpy as np
from preprocess import *
import pymorphy2
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

morph = pymorphy2.MorphAnalyzer()

data_dir = '/ABSA/cleand_query_output_csv/'

test_file = '0a5c0a4c-36f7-46c4-9f13-91f52ba45ea5'

file_path = os.path.join(data_dir, test_file)



def get_text(filepath):
    df = pd.read_csv(filepath)
    # replace the NAN with empty string.
    df = df.replace(np.nan, '', regex=True)
    df.text = df.text.astype(str)
    df.title = df.title.astype(str)

    # pandas dataframe. words = title+text.
    df["words"] = df.title + ' ' + df.text
    # filter the english content.
    en_df = df[df.lang == 'en']
    en_df = en_df[en_df.words.notnull()]

    docs = en_df.words.to_list()  # => data
    return docs


def preprocess_text(docs):
    texts = []
    for doc in docs :
        sents = sent_tokenize(doc)
        print('sentized:')
        parsed_sents = [' '.join(parseSentence(sent)) for sent in sents]
        print(parsed_sents)
        print('*'*5)
        texts.extend(parsed_sents)
    return texts

if __name__ == '__main__':
    docs = get_text(file_path)
    texts = preprocess_text(docs)
    print(texts)