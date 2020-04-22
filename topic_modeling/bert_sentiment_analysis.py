from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from sklearn.model_selection import train_test_split

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

from utils.timer import Timer
from bert_sentiment_model import model_fn_builder

def review_sent_tokenize(review):
    """
    Sentenized the reviews.
    :param review:
    :return:
    """
    countries = ['Netherlands', 'Italy', 'France', 'Austria', 'Spain', 'United Kingdom']
    locations = ['Amsterdam', 'Barcelona', 'Paris', 'Vienna', 'Milan']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December', 'Jan.', 'Jan', 'Feb.', 'Feb', 'Mar.', 'Mar', 'Apr.', 'Apr', 'Jun.', 'Jun',
                    'Jul', 'Jul.', 'Aug.', 'Aug', 'Sep', 'Sep.', 'Oct.', 'Oct', 'Nov.', 'Nov',
                    'Dec.', 'Dec']
    week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
           'Mon.', 'Tue.', 'Wed.', 'Thu.', 'Fri.', 'Sat.', 'Sun.']

    doc = review.strip().split()
    upper_idx = []
    for idx, word in enumerate(doc):
        if word.istitle():
            if word not in countries + ['United', 'Kingdom'] + locations + months + week:
                upper_idx.append(idx)

    sents = []
    if len(doc) > 1:
        if len(upper_idx) > 1:
            for ind, indice in enumerate(upper_idx[:-1]):
                sent = doc[indice:upper_idx[ind + 1]]
                if len(sent) > 2:
                    sents.append(' '.join(sent))
            sent = doc[upper_idx[-1]:]
            if len(sent) > 2:
                sents.append(' '.join(sent))
        else:
            sents.append(' '.join(doc))

    return sents


def get_hotel_data():
    df = pd.read_csv('DATASET/Hotel_Reviews.csv')

    neg_reviews = []
    pos_reviews = []
    for hotel_name, pos_review, neg_review in zip(df['Hotel_Name'], df['Positive_Review'], df['Negative_Review']):

        hotel_name_strip = hotel_name.replace(' ', '[HOTEL]')
        if neg_review.strip() != 'No Negative':
            neg_review = neg_review.replace(hotel_name, hotel_name_strip)
            neg_reviews += review_sent_tokenize(neg_review)

        if pos_review.strip() != 'No Positive':
            pos_review = pos_review.replace(hotel_name, hotel_name_strip)
            pos_reviews += review_sent_tokenize(pos_review)
    pos_labels = [1 for _ in range(len(pos_reviews))]
    pos_df = pd.DataFrame(list(zip(pos_reviews, pos_labels)), columns=['sentence', 'label'])

    neg_labels = [0 for _ in range(len(neg_reviews))]
    neg_df = pd.DataFrame(list(zip(neg_reviews, neg_labels)), columns=['sentence', 'label'])

    merged_df = pd.concat([pos_df, neg_df])

    merged_df = merged_df.sample(frac=1).reset_index(drop=True)
    merged_df.to_csv('DATASET/reviews.csv')


def get_data():
    df = pd.read_csv('DATASET/Hotel_Reviews.csv')
    neg_reviews = df[df['Negative_Review'] != 'No Negative']['Negative_Review']
    pos_reviews = df[df['Positive_Review'] != 'No Positive']['Positive_Review']
    
    pos_labels = [1 for _ in range(len(pos_reviews))]
    pos_df = pd.DataFrame(list(zip(pos_reviews, pos_labels)), columns=['sentence', 'label'])

    neg_labels = [0 for _ in range(len(neg_reviews))]
    neg_df = pd.DataFrame(list(zip(neg_reviews, neg_labels)), columns=['sentence', 'label'])
    
    merged_df = pd.concat([pos_df, neg_df])
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)

    train, test_dev = train_test_split(merged_df, test_size=0.3)
    test, dev = train_test_split(test_dev, test_size=0.5)

    pos_train = train[train['label'] == 1]
    neg_train = train[train['label'] == 0]
    pos_test = test[test['label'] == 1]
    neg_test = test[test['label'] == 0]
    pos_dev = dev[dev['label'] == 1]
    neg_dev = dev[dev['label'] == 0]
    print('positive train samples: ', len(pos_train))
    print('negative train samples: ', len(neg_train))
    print('positive test samples: ', len(pos_test))
    print('negative test samples: ', len(neg_test))
    print('positive dev samples: ', len(pos_dev))
    print('negative dev samples: ', len(neg_dev))

    train.to_csv('DATASET/train.csv')
    dev.to_csv('DATASET/dev.csv')
    test.to_csv('DATASET/test.csv')
    return train, dev, test



def create_tokenizer_from_hub_module(BERT_MODEL_HUB):
    """Get the vocab file and casing info from the Hub module."""


    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.compat.v1.Session() as sess:
              vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                    tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=do_lower_case)


def load_train_test_data(filepath):
    df = pd.read_csv(filepath)
    train, test = train_test_split(df, test_size=0.2)
    pos_train = train[train['label']==1]
    neg_train = train[train['label']==0]
    pos_test = test[test['label'] == 1]
    neg_test = test[test['label'] == 0]

    print('positive train samples: ', len(pos_train))
    print('negative train samples: ', len(neg_train))
    print('positive test samples: ', len(pos_test))
    print('negative test samples: ', len(neg_test))

    return train, test

def load_train_test_dev_data(filepath):
    df = pd.read_csv(filepath)
    train, test_dev = train_test_split(df, test_size=0.3)
    test, dev = train_test_split(test_dev, test_size=0.5)
    pos_train = train[train['label']==1]
    neg_train = train[train['label']==0]
    pos_test = test[test['label'] == 1]
    neg_test = test[test['label'] == 0]
    pos_dev = dev[dev['label'] == 1]
    neg_dev = dev[dev['label'] == 0]

    print('positive train samples: ', len(pos_train))
    print('negative train samples: ', len(neg_train))
    print('positive test samples: ', len(pos_test))
    print('negative test samples: ', len(neg_test))
    print('positive dev samples: ', len(pos_dev))
    print('negative dev samples: ', len(neg_dev))

    train.to_csv('DATASET/train.csv')
    dev.to_csv('DATASET/dev.csv')
    test.to_csv('DATASET/test.csv')

    return train, test, dev


def train_model(train, test, BERT_MODEL_HUB):
    # create examples from dataframes.
    train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                 text_a=x['sentence'],
                                                                                 text_b=None,
                                                                                 label=x['label']), axis=1)

    test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                               text_a=x['sentence'],
                                                                               text_b=None,
                                                                               label=x['label']), axis=1)
    tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_HUB)

    MAX_SEQ_LENGTH=128
    label_list = [0, 1]


    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH,
                                                                      tokenizer)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH,
                                                                     tokenizer)

    # Compute train and warmup steps from batch size
    # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-3
    NUM_TRAIN_EPOCHS = 5
    # Warmup is a period of time where hte learning rate
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100

    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
    run_config = tf.estimator.RunConfig(
        model_dir='bert_model/',
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(
        num_labels=len(label_list),
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})

    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)


    print(f'Beginning Training!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)

    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    estimator.evaluate(input_fn=test_input_fn, steps=None)



if __name__ == '__main__':
    timer = Timer()
    timer.start()
    get_data()
    # BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    # train_model(train, test , BERT_MODEL_HUB)



    timer.stop()
