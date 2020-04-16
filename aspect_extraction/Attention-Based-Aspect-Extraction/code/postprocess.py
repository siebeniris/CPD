from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from ast import literal_eval
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

def parse_xml(output):

    with open(output, 'rt') as file:
        tree = ElementTree.parse(file)
        root = tree.getroot()
    for c in range(4209):
        path_ = './/sentence[@id="{}"]'.format(str(c))
        sentence = root.find(path_)
        orig_text = sentence.attrib.get('orig')

        indices_sentence = literal_eval(sentence.attrib.get('indices'))

        noun_phrases = literal_eval(sentence.attrib.get('nounIndices'))

def postprocess_xml_output(output, sentiment_file):
    sentiments = load_sentiment_terms(sentiment_file)
    with open(output, 'rt') as file:
        tree = ElementTree.parse(file)
        root = tree.getroot()

    count = 0
    for sentence in root.iter('sentence'):

        id = sentence.attrib.get('id')
        lemmas_indices = literal_eval(sentence.attrib.get('indices'))
        nounIndices = literal_eval(sentence.attrib.get('nounIndices'))
        nounPhrases = literal_eval(sentence.attrib.get('nounPhrases'))
        orig = sentence.attrib.get('orig')
        lemmas = sentence.text
        aspectTerms = sentence.find('aspectTerms')
        ind_word_weight = literal_eval(aspectTerms.attrib.get('ind_word_weight'))

        aspectTerms_phrases =[]
        for offsets_lemma, aspect_term, weight in ind_word_weight:
            # check if the aspect_term is a noun phrase, if it is, get the indices.
            for idx, noun_phrase in enumerate(nounPhrases):
                # if aspect_term in noun_phrase:
                offsets_noun_phrase = nounIndices[idx]
                if offsets_lemma[0] >= offsets_noun_phrase[0] and offsets_lemma[1] <= offsets_noun_phrase[1]:
                    aspectTerms_phrases.append((offsets_noun_phrase, noun_phrase))
            if not any([offsets_lemma[0] >= offsets_noun_phrase[0] and offsets_lemma[1] <= offsets_noun_phrase[1]
                        for offsets_noun_phrase, _ in list(set(aspectTerms_phrases))]):
                aspectTerms_phrases.append((offsets_lemma, aspect_term))

        sorted_aspects = sorted(list(set(aspectTerms_phrases)), key=lambda x:x[0])
        processed_aspects =[]
        for offsets, aspect_term in sorted_aspects:
            orig_term = orig[offsets[0]:offsets[1]]
            print(orig_term)
            starting = aspect_term.split()[0]
            if starting in sentiments:
                new_starting = offsets[0]+len(starting)+1
                new_term = orig[new_starting: offsets[1]]
                processed_aspects.append(((new_starting, offsets[1]), new_term))
            else:
                processed_aspects.append((offsets, orig_term))



        print(id, orig)
        print(lemmas, lemmas_indices)
        print(nounPhrases, nounIndices)
        print(ind_word_weight)
        print(sorted_aspects)
        print(processed_aspects)
        print("*"*8)
        count += 1


def get_phrases():
    import itertools
    sentence = 'A stunning weekend getaway at Phinda Mountain Lodge .'
    offsets = [(11,18), (19, 26), (30, 36), (37, 45), (46,51)]

    offset = 0
    offset_dict = dict()
    tokens = word_tokenize(sentence)
    print(tokens)
    for idx, word in enumerate(tokens):
        offset_dict[ (offset, offset+len(word))] = idx
        offset += len(word)+1

    print(offset_dict)
    indices = [offset_dict[x] for x in offsets]
    for group in consecutive_groups(indices):
        index=list(group)
        start, end = index[0], index[-1]
        print(tokens[start:end])




if __name__ == '__main__':
    output = 'output.xml'
    # postprocess_xml_output(output, 'sentiments.txt')
    get_phrases()