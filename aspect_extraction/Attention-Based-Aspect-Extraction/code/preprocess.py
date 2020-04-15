import codecs
import argparse
import os
import glob
import csv
import string

from joblib import Parallel, delayed

from textblob import TextBlob, Word
from textblob.np_extractors import ConllExtractor, FastNPExtractor

import spacy

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.dom import minidom

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

nlp =spacy.load('en_core_web_sm')

# https://pymotw.com/2/xml/etree/ElementTree/create.html
def prettify(elem):
    """Return a pretty-printed XML string for the Element.
        """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def load_sentiment_terms(file):
    # pos, neg
    # adjectives = ['JJ','JJR','JJS']
    terms = []
    with open(file) as input:
        for line in input:
            if not line.startswith(';'):
                terms.append(line.strip())
    return list(set(terms))


# https://www.clips.uantwerpen.be/pages/mbsp-tags
def parseTextBlob(line):
    stop = stopwords.words('english')
    noun_tags = ['NN', 'NNP', 'NNPS', 'NNS']
    verbs = ['VB', 'VBZ', 'VBP', 'VBN', 'VBD', 'VBG']

    # have a testblob of the line.
    # wiki = TextBlob(line)
    # tags = wiki.tags
    tags = line.tags

    # if the tag is noun, it should not be corrected.
    # b_tags = [(x.correct(), y) if y not in noun_tags else (x,y) for x,y in tags]
    # correct_line = ' '.join([b for b,_ in b_tags])

    # corrected = TextBlob(line, np_extractor=ConllExtractor())
    noun_phrases = line.noun_phrases
    lowered = line.lower()
    # eliminate stop words in the sentence.
    b_tags = [(x.lower(), y) for x, y in tags if x.lower() not in stop]
    noun_indices = []
    offset_noun = 0
    for noun in noun_phrases:
        noun = noun.lower()
        try:
            pos = lowered.index(noun, offset_noun)
            noun_indices.append((pos, pos+len(noun)))
            offset_noun = pos+len(noun)
        except Exception:
            zero = noun.split()[0]
            pos = lowered.index(zero, offset_noun)
            noun_indices.append((pos, pos+len(noun)))
            offset_noun = pos +len(noun)

    # indices for text for training.
    indices = []
    offset = 0
    for word, _ in b_tags:
        ind = lowered.index(word, offset)
        indices.append((ind, ind+len(word)))
        offset = ind+len(word)

    # lemmatize words.
    b = [Word(x).lemmatize('v') if y in verbs else Word(x).lemmatize() for x,y in b_tags]
    lemmas = ' '.join(b)
    return lemmas, indices, noun_phrases, noun_indices
    # if len(indices) > 0:
    #     sent_ = SubElement(review, 'sentence', {'id': str(count), 'orig': line,
    #                                             'indices': str(indices),
    #                                             'nounPhrases': str(noun_phrases),
    #                                             'nounIndices': str(noun_indices)})
    #     sent_.text = lemmas
    #     count += 1
    #     print(line, noun_phrases, count)
    #
    #     return sent_, count

def parseSentence(line):
    # delete stop words and lemmatize.
    lmtzr = WordNetLemmatizer()
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())

    text_rmstop = [i for i in text_token if i not in stop]
    offset=0
    indices =[]
    for word in text_rmstop:
        ind = line.lower().index(word, offset)
        indices.append((ind, ind+len(word)))
        offset += len(word)

    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem, indices


def spacyParseSentence(line):
    line = line.lower()
    doc = nlp(line)
    noun_phrases =[chunk.text for chunk in doc.noun_chunks]
    noun_phrases_indices = []
    offset = 0
    for phrase in noun_phrases:
        ind = line.index(phrase, offset)
        noun_phrases_indices.append((ind, ind+len(phrase)))
        offset += len(phrase)
    print(noun_phrases)
    return noun_phrases, noun_phrases_indices



# https://pymotw.com/2/xml/etree/ElementTree/create.html
# https://stackoverflow.com/a/3605831
def generate_test_xml_from_ty(filepath, output):
    """
    Generate xml file for test content to infer.
    :param filepath: The cleand query csv file.
    :param output: directory of output file.
    :return: prettified xml formatted output
    """
    root = Element('reviews')
    # load the sentiments terms.

    sentiments = load_sentiment_terms('sentiments.txt')

    count = 0
    with open(filepath, 'rt') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # skip header
        for row in csvreader:
            uid, date, score, rec_rate,lang, title, text = row
            review = SubElement(root, 'review', {'uid': uid,
                                                 'date': date,
                                                 'score': score,
                                                 'recommendation_rate': rec_rate,
                                                 'lang': lang})
            if lang == 'en':  # only get english sentences
                if title:
                    if title[-1] in string.punctuation:
                        original_text = title+' ' +text
                    else:
                        original_text = title+'.'+text
                else:
                    original_text = text

                if len(original_text.strip()) > 0:

                    # orig_text = SubElement(review, 'original_text')
                    # orig_text.text = original_text

                    sents = TextBlob(original_text, np_extractor=ConllExtractor()).sentences
                    # if sents:
                    #     for sent in sents:
                    #          sent_, count = parseTextBlob(review, count, sent)
                    # Parallel(n_jobs=-1)(map(delayed(parseTextBlob),[[review, idx, sent]for idx, sent in enumerate(sents)]))
                    for sent in sents:
                        print(sent)
                        lemmas, indices, noun_phrases, noun_indices = parseTextBlob(sent)
                        # tokens, indices = parseSentence(sent)
                        # noun_phrases, noun_indices = spacyParseSentence(sent)
                        if len(indices) > 0:
                            sent_ = SubElement(review, 'sentence', {'id': str(count), 'orig': sent,
                                                                    'indices': str(indices),
                                                                    'nounPhrases': str(noun_phrases),
                                                                    'nounIndices':str(noun_indices)})
                            sent_.text = lemmas
                            count += 1
    tree = ElementTree.ElementTree(root)
    tree.write(output)




def generate_train_files_from_ty(dir_path, output_path, num_sents=1000000):
    output_file = os.path.join(output_path, 'train.txt')
    output = open(output_file, 'w')
    count = 0
    for filepath in glob.glob(dir_path+'/*'):
        if os.path.isfile(filepath):
            # read csv file into Dataframe
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

            docs = en_df.words.to_list()
            for doc in docs:
                sents = sent_tokenize(doc)
                for sent in sents:
                    tokens, _ = parseSentence(sent)
                    if count > num_sents:
                        break
                    else:
                        if len(tokens) > 1:
                            output.write(' '.join(tokens)+'\n')
                        count += 1
                        print('sentence:{}'.format(count), end='\r')


def preprocess_test_ty(input, domain):
    outfile = os.path.join('../preprocessed_data', domain, 'test.txt' )
    f1 = codecs.open(input, 'r', 'utf-8')
    out1 = codecs.open(outfile, 'w', 'utf-8')

    df = pd.read_csv(f1)
    # replace the NAN with empty string.
    df = df.replace(np.nan, '', regex=True)
    df.text = df.text.astype(str)
    df.title = df.title.astype(str)

    # pandas dataframe. words = title+text.
    df["words"] = df.title + ' ' + df.text
    # filter the english content.
    en_df = df[df.lang == 'en']
    en_df = en_df[en_df.words.notnull()]

    docs = en_df.words.to_list()
    for doc in docs:
        sents = sent_tokenize(doc)
        for sent in sents:
            tokens, _ = parseSentence(sent)

            if len(tokens) > 1:
                    out1.write(' '.join(tokens) + '\n')


if __name__ == "__main__":
    input ='/home/yiyi/Documents/masterthesis/CPD/data/cleand_query_output_csv'
    output = '../preprocessed_data/ty'
    test_filename = '0a5c0a4c-36f7-46c4-9f13-91f52ba45ea5'
    test_file = os.path.join(input, test_filename)
    # preprocess_test_ty(test_file, 'ty')
    # generate_train_files_from_ty(input, output)
    output_dir = '/home/yiyi/Documents/masterthesis/CPD/data/aspect_extraction'
    output_file = os.path.join(output_dir, test_filename)
    generate_test_xml_from_ty(test_file, output_file+'.xml')


