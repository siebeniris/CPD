# Hilton miami downtown
# 2018-01-17T16:25:00+02:00
# 527 rooms
# 35 million renovation
# mammoth shopping complex, movie theater , 
# guest rooms, meeting space, restuarant, lobby, loading dock, park ,
import os

import spacy
import pandas as pd
import rootpath

import utils.preprocess as preprocess
from utils.timer import Timer
from keras_spacy_SA import *



def get_prediction_for_one_file(nlp, inputfile, outputfile):
    uids, lemmas, sents, dates, scores = preprocess.get_data(inputfile)
    sentiments = []
    # deduplicate the spaces in sentence.
    lowerd = [sent.lower() for sent in sents]
    for doc in nlp.pipe(lowerd):
        sentiments.append(doc.sentiment)

    sent_df = pd.DataFrame(zip(uids, lemmas, sents, sentiments, dates, scores),
                           columns=['uid', 'lemma', 'sentence', 'sentiment', 'date', 'score'])
    sent_df.to_csv(outputfile, index=False)


if __name__ == '__main__':
    nlp = spacy.load("en_vectors_web_lg")
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    ##### Applying Sentiment Analyser
    nlp.add_pipe(SentimentAnalyser.load('2lstm_model/', nlp, max_length=128))

    # root_dir = rootpath.detect()
    root_dir = '/home/yiyi/Documents/masterthesis/CPD'
    input_dir = os.path.join(root_dir, 'data/cleand_query_output_csv')
    output_dir = os.path.join(root_dir, 'data/sentiment_analysis/results')

    timer = Timer()

    for filename in os.listdir(input_dir):

        inputfilepath = os.path.join(input_dir, filename)
        outputfilepath = os.path.join(output_dir, filename)
        if os.path.isfile(inputfilepath) and not os.path.exists(outputfilepath):
            timer.start()
            get_prediction_for_one_file(nlp, inputfilepath, outputfilepath)
            print('processed file ', filename)
            timer.stop()
