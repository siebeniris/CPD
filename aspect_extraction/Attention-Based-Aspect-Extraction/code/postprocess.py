from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from ast import literal_eval
from nltk.corpus import wordnet as wn
from preprocess import load_sentiment_terms

output= 'output.xml'
with open(output, 'rt') as file:
    tree = ElementTree.parse(file)
    root = tree.getroot()

path_ = './/sentence[@id="{}"]'.format(str(c))
sentence = root.find(path_)
orig_text = sentence.attrib.get('orig')

indices_sentence = literal_eval(sentence.attrib.get('indices'))

noun_phrases = literal_eval(sentence.attrib.get('nounIndices'))
