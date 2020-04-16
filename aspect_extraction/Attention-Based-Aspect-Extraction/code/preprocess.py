import codecs
import os
import glob
import csv
import string
import regex
from nltk.corpus import wordnet
from timer import Timer

from textblob import TextBlob, Word
from textblob.np_extractors import ConllExtractor, FastNPExtractor
from contractions import contractions_dict
import spacy

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.dom import minidom

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

nlp =spacy.load('en_core_web_sm')

# https://pymotw.com/2/xml/etree/ElementTree/create.html
def prettify(elem):
    """Return a pretty-printed XML string for the Element.
        """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


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

def parseSentence(line):

    # delete stop words and lemmatize.
    print('original:', line)
    lmtzr = WordNetLemmatizer()

    stop = stopwords.words('english')
    # text_token = CountVectorizer().build_tokenizer()(line.lower())
    expanded = expand_contractions(line, contractions_dict)
    text_token = [remove_repeated_characters(word) for word in word_tokenize(expanded.lower())]

    print(text_token)
    offset =0
    offset_dict = dict()
    lemmas= []
    lemmas_index=[]
    for idx, word in enumerate(text_token):
        offset_dict[(offset, offset+len(word))] = idx
        if word not in stop:
            lemmas.append(lmtzr.lemmatize(word))
            lemmas_index.append(idx)
        offset += len(word)+1


    return ' '.join(lemmas), lemmas_index, offset_dict


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
    timer = Timer()
    timer.start()
    root = Element('reviews')
    # load the sentiments terms.

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
                    sents = sent_tokenize(original_text.replace('.', '. '))
                    for sent in sents:
                        lemmas, indices, offset_dict = parseSentence(sent)

                        if len(indices) > 0:
                            sent_ = SubElement(review, 'sentence', {'id': str(count), 'orig': sent,
                                                                    'indices': str(indices),
                                                                    'offsetDict': str(offset_dict)})
                            sent_.text = lemmas
                            count += 1
                            print('sentence:{}'.format(count), end='\r')

    tree = ElementTree.ElementTree(root)
    tree.write(output)
    timer.stop()




def generate_train_files_from_ty(dir_path, output_path, num_sents=1000000):
    timer = Timer()
    timer.start()
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
                    tokens, _, _ , _ = parseSentence(sent)
                    if count > num_sents:
                        break
                    else:
                        if len(tokens) > 1:
                            output.write(' '.join(tokens)+'\n')
                        count += 1
                        print('sentence:{}'.format(count), end='\r')

    timer.stop()


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
            tokens, _, _, _ = parseSentence(sent)

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
    line = "You can't go wrong with Phinda and we are already looking to book our next adventure with &beyond!"
    print(parseSentence(line))


