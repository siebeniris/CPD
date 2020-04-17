import codecs
import re
import operator
import os
from xml.etree import ElementTree
from xml.dom import minidom
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


def is_number(token):
    return bool(num_regex.match(token))


def create_vocab(domain, maxlen=0, vocab_size=0):
    # assert domain in {'restaurant', 'beer'}

    # source = '../preprocessed_data/' + domain + '/train.txt'
    source = os.path.join('../preprocessed_data', domain, 'train.txt')

    total_words, unique_words = 0, 0
    word_freqs = {}
    top = 0

    fin = codecs.open(source, 'r', 'utf-8')
    for line in fin:
        words = line.split()
        if maxlen > 0 and len(words) > maxlen:
            continue

        for w in words:
            if not is_number(w):
                try:
                    word_freqs[w] += 1
                except KeyError:
                    unique_words += 1
                    word_freqs[w] = 1
                total_words += 1

    print(' %i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    # vocab
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = len(vocab)
    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print ('  keep the top %i words' % vocab_size)

    # Write (vocab, frequence) to a txt file
    vocab_file = codecs.open('../preprocessed_data/%s/vocab' % domain, mode='w', encoding='utf8')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocab:
        if index < 3:
            vocab_file.write(word + '\t' + str(0) + '\n')
            continue
        vocab_file.write(word + '\t' + str(word_freqs[word]) + '\n')
    vocab_file.close()

    return vocab


def read_dataset(domain, phase, vocab, maxlen):
    # assert domain in {'restaurant', 'beer'}
    assert phase in {'train', 'test'}
    source = os.path.join('../preprocessed_data', domain, phase+'.txt')
    # source = '../preprocessed_data/' + domain + '/' + phase + '.txt'
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    data_x = []

    fin = codecs.open(source, 'r', 'utf-8')
    for line in fin:
        words = line.strip().split()
        if maxlen > 0 and len(words) > maxlen:
            words = words[:maxlen]
        if not len(words):
            continue

        indices = []
        for word in words:
            # if it is a number.
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            # if it is unknown word.
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data_x.append(indices)
        if maxlen_x < len(indices):
            maxlen_x = len(indices)

    print('   <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    return data_x, maxlen_x


# https://pymotw.com/2/xml/etree/ElementTree/create.html
def prettify(elem):
    """Return a pretty-printed XML string for the Element.
        """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")



def read_dataset_ty(filepath, vocab, maxlen):
    # assert domain in {'restaurant', 'beer'}
    with open(filepath, 'rt') as file:
        tree = ElementTree.parse(file)

    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    data_x = []

    for line in tree.iter('sentence'):
        words = line.text.strip().split()
        if maxlen > 0 and len(words) > maxlen:
            words = words[:maxlen]
        if not len(words):
            continue

        indices = []
        for word in words:
            # if it is a number.
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            # if it is unknown word.
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data_x.append(indices)
        if maxlen_x < len(indices):
            maxlen_x = len(indices)

    print('   <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    return data_x, maxlen_x

def get_train_data(domain, vocab_size=0, maxlen=0):
    vocab = create_vocab(domain, maxlen, vocab_size)
    train_x, train_maxlen = read_dataset(domain, 'train', vocab, maxlen)
    return vocab, train_x, train_maxlen

def get_test_data(domain, maxlen=0):
    vocab_file = os.path.join('../preprocessed_data', domain, 'vocab')
    vocab_ = codecs.open(vocab_file, 'r', 'utf-8')
    vocab={}
    for line in vocab_:
        word, freq = line.strip().split('\t')
        vocab[word]=freq
    test_x, test_maxlen = read_dataset(domain, 'test', vocab, maxlen)
    return vocab, test_x, test_maxlen



def get_data(domain, vocab_size=0, maxlen=0):
    print('Reading data from ', domain)
    print(' Creating vocab ...')
    vocab = create_vocab(domain, maxlen, vocab_size)
    print(' Reading dataset ...')
    print('  train set')
    train_x, train_maxlen = read_dataset(domain, 'train', vocab, maxlen)
    print('  test set')
    test_filename = '0a5c0a4c-36f7-46c4-9f13-91f52ba45ea5'
    output_dir = '/ABSA/aspect_extraction'
    test_xml = os.path.join(output_dir, test_filename + '.xml')
    test_x, test_maxlen = read_dataset_ty(test_xml, vocab, maxlen)
    maxlen = max(train_maxlen, test_maxlen)
    return vocab, train_x, test_x, maxlen


def get_data_ty(domain,test_path, vocab_size=0, maxlen=0):
    print('Reading data from ', domain)
    print(' Creating vocab ...')
    vocab = create_vocab(domain, maxlen, vocab_size)
    print(' Reading dataset ...')
    print('  train set')
    train_x, train_maxlen = read_dataset(domain, 'train', vocab, maxlen)
    print('  test set')

    test_x, test_maxlen = read_dataset_ty(test_path, vocab, maxlen)
    maxlen = max(train_maxlen, test_maxlen)
    return vocab, train_x, test_x, maxlen



if __name__ == "__main__":
    # vocab, train_x, test_x, maxlen = get_data('restaurant')
    # print('train_x sample:', train_x[:10])
    # print(len(train_x))
    # print(len(test_x))
    # print(maxlen)
    # vocab, test_x, test_maxlen = get_test_data('ty')
    # print(vocab['<unk>'])
    # print(test_maxlen)
    #
    # _, train, train_maxlen= get_train_data('ty')
    # print(train_maxlen)
    test_filename = '0a5c0a4c-36f7-46c4-9f13-91f52ba45ea5'
    output_dir = '/ABSA/aspect_extraction'
    test_xml = os.path.join(output_dir, test_filename+'.xml')
    read_test_xmls(test_xml)




