### evaluate the inter annotator aggrement.
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa


def load_annotations(file='data_processed/annotations.json'):
    with open(file) as reader:
        annotations = json.load(reader)
    return annotations


def get_annotations_dict(annotations):
    """
    Get annotations_dict.
    :param annotations: loaded from database
    :return: dictionary of annotations.
    """
    new_annotations = defaultdict(list)

    # filId: [list of annotations from different annotators]
    for a in annotations:
        idx = int(a['fileId'])
        new_annotations[idx].append(a)

    annotations_dict = defaultdict(list)
    # get the most recent annotations for each annotator!
    for idx, al in new_annotations.items():
        adict = defaultdict(list)

        for a in al:
            username = a['username']
            adict[username].append(a)

            for username, a in adict.items():
                if len(a) > 1:
                    sorted_a = sorted(a, key=lambda i: i['date'])
                    # get the most recent annotation from one user.
                    annotations_dict[idx].append(sorted_a[-1])
                else:
                    annotations_dict[idx].append(a[0])

    return annotations_dict


def prepare_dict(annotation_dict):
    annotator_dict_cpt = defaultdict(dict)  # annotator: annotaiton
    annotator_dict_sent_sentiment = defaultdict(dict)
    annotator_dict_sent_aspect = defaultdict(dict)

    # fileId: int
    for fileId, annotations in annotation_dict.items():

        for a in annotations:
            annotation = a['annotation']
            username = a['username']
            if a['cptAnswer']:
                cpts = a['cptAnswer']

                for nr, cpt in cpts.items():
                    id_nr = str(fileId) + '_' + nr
                    annotator_dict_cpt[username][id_nr] = cpt

            if username != 'yiyi':

                for anno in annotation:
                    id_ = anno['id']

                    yes_aspect_ = anno.get('yes_aspect', False)
                    yes_sentiment_ = anno.get('yes_sentiment', False)

                    id_sentid = str(fileId) + '_' + str(id_)

                    if not yes_aspect_:
                        annotator_dict_sent_aspect[username][id_sentid] = 0
                    if not yes_sentiment_:
                        annotator_dict_sent_sentiment[username][id_sentid] = 1

                    if yes_aspect_:
                        annotator_dict_sent_aspect[username][id_sentid] = 2
                    if yes_sentiment_:
                        annotator_dict_sent_sentiment[username][id_sentid] = 3

    return annotator_dict_cpt, annotator_dict_sent_sentiment, annotator_dict_sent_aspect


def get_cohen_kappa(df):
    """
    Calculate the cohen's kappa. (between two annotators)
    :param df:
    :return:
    """
    count_agree = 0
    count_all = 0

    anno1, anno2 = [], []
    if 'yiyi' in df.columns:
        df = df.drop(columns=['yiyi'])
    for idx, row in df.iterrows():
        row_na = row.dropna().tolist()
        if len(row_na) == 2:
            a1, a2 = row_na
            anno1.append(a1)
            anno2.append(a2)
            if a1 == a2:
                count_agree += 1
            count_all += 1
    score = cohen_kappa_score(anno1, anno2)

    print('df len:', len(df))
    print('annotated twice len:', len(anno1))
    print('cohens score:', score)
    print('agreement percentage:', count_agree / count_all)
    print('***' * 20)


def get_fleiss_kappa(df, cpt=False):
    print('calculate fleiss kappa')
    if cpt:
        print('total length:', len(df))
        df = df[df['Total'] == 2]
        print('total annotators 2 cpt len:', len(df))
        categories = ['a', 'b', 'c']
        df = df[categories]
        M = df.to_numpy()

        fless = fleiss_kappa(M)
        print('fless kappa cpt:', fless)
    else:
        df['total-aspect'] = df['no-aspect'] + df['yes-aspect']
        df['total-sentiment'] = df['no-sentiment'] + df['yes-sentiment']
        print('all df len:', len(df))
        df_sent = df[(df['no-aspect'] == 0) & (df['total-sentiment']==2)]

        print('no no-aspect at least 2 annotators:', len(df_sent))
        df_sentiment = df_sent[['yes-sentiment', 'no-sentiment']]

        df_as = df[(df['no-sentiment'] == 0) & (df['total-aspect']==2)]
        print('no no-sentiment at least 2 annotators:', len(df_as))
        df_aspect = df_as[['yes-aspect', 'no-aspect']]

        M_sentiment = df_sentiment.to_numpy()
        M_aspect = df_aspect.to_numpy()
        fless_sentiment = fleiss_kappa(M_sentiment)
        fless_aspect = fleiss_kappa(M_aspect)
        print('fless kappa sentiment:', fless_sentiment)
        print('fless kappa aspect', fless_aspect)
    print('*' * 20)


def sent_agreement(df):
    ### yes-aspect=2
    print('*' * 20)

    print('all df len:', len(df))
    df_sentiment = df[(df['no-aspect'] == 0)]
    df_aspect = df[(df['no-sentiment'] == 0)]

    agree_sentiment = df_aspect[(df_aspect['yes-sentiment'] == 2) | (df_aspect['no-sentiment'] == 2)]
    print(len(agree_sentiment), len(df_aspect), len(df))
    print('agree sentiment:', len(agree_sentiment) / len(df), len(agree_sentiment) / len(df_aspect))

    agree_aspect = df_sentiment[(df_sentiment['yes-aspect'] == 2) | (df_sentiment['no-aspect'] == 2)]
    print(len(agree_aspect), len(df_sentiment), len(df))
    print('agree sentiment:', len(agree_aspect) / len(df), len(agree_aspect) / len(df_sentiment))


def human_performance_cpt(df):
    print('*' * 40)

    print('Human performance, how often one annotator agrees with the gold label? (change point candidates)')
    print('cpt total length', len(df))
    cpt_df = df.replace({'a': 0, 'b': 1, 'c': 1})
    cpt_df['least2'] = cpt_df.apply(lambda x: len(x.dropna().tolist()) > 1, axis=1)
    cpt_df = cpt_df[cpt_df['least2'] == True]
    print('at least annotated twice:', len(cpt_df))

    cpt_df['gold'] = cpt_df.apply(lambda x: np.bincount(x.dropna()).argmax(), axis=1)

    annotators_dict = {}  # how many times annotators agree with the gold label
    annotators = ['Andrei', 'Liviu', 'Alina', 'Oana', 'Edita', 'Silviu', 'Mada', 'Meli', 'yiyi']
    agreed_array = []
    agreed_total = 0
    total_annotated = 0
    for annotator in annotators:
        cpt_df_anno = cpt_df[[annotator, 'gold']]
        cpt_df_anno.dropna(inplace=True)
        all = len(cpt_df_anno)
        agreed = len(cpt_df[cpt_df[annotator] == cpt_df['gold']])
        annotators_dict[annotator] = {
            'all': all,
            'agreed': agreed,
            'percentag': agreed / all
        }
        agreed_array.append(agreed / all)
        agreed_total += agreed
        total_annotated += all
    print('in total annotators:', len(annotators))
    print('avg agreed:', agreed_total / len(annotators))
    print('avg annotated:', total_annotated / len(annotators))
    print(annotators_dict)
    print(np.mean(agreed_array))


def get_gold_label_sent(row, typ):
    if typ == 'sentiment':
        row_list = [int(x) for x in row.dropna().tolist()]

        if row_list[:-1] == [3, 3]:
            return 3
        else:
            return 1
    # both said yes.
    if typ == 'aspect':
        row_list = [int(x) for x in row.dropna().tolist()]
        if row_list[:-1] == [2, 2]:
            return 2
        else:
            return 0


def human_performance_sent(df, typ):
    print('*' * 40)
    print('Human performance, how often one annotator agrees with the gold label? (sentence annotator)')
    print('sentences total length', len(df))

    df['least2'] = df.apply(lambda x: len(x.dropna().tolist()) > 1, axis=1)
    df = df[df['least2'] == True]
    print('at least annotated twice:', len(df))

    df['gold'] = df.apply(lambda x: get_gold_label_sent(x, typ), axis=1)

    annotators_dict = {}  # how many times annotators agree with the gold label
    annotators = ['Andrei', 'Liviu', 'Alina', 'Oana', 'Edita', 'Silviu', 'Mada', 'Meli', ]
    agreed_array = []
    agreed_total = 0
    total_annotated = 0
    for annotator in annotators:
        df_anno = df[[annotator, 'gold']]
        df_anno.dropna(inplace=True)
        all = len(df_anno)
        agreed = len(df_anno[df_anno[annotator] == df_anno['gold']])
        annotators_dict[annotator] = {
            'all': all,
            'agreed': agreed,
            'percentag': agreed / all
        }
        agreed_array.append(agreed / all)
        agreed_total += agreed
        total_annotated += all
    print('in total annotators:', len(annotators))
    print('avg agreed:', agreed_total / len(annotators))
    print('avg annotated:', total_annotated / len(annotators))
    print(annotators_dict)
    print(np.mean(agreed_array))


if __name__ == '__main__':
    # annotations = load_annotations()
    # annotations_dict = get_annotations_dict(annotations)
    # annotator_dict_cpt, annotator_dict_sentiment, annotator_dict_aspect = prepare_dict(annotations_dict)
    #
    # annotator_sent_sentiment_df = pd.DataFrame.from_dict(annotator_dict_sentiment)
    # annotator_sent_aspect_df = pd.DataFrame.from_dict(annotator_dict_aspect)
    #
    # annotator_cpt_df = pd.DataFrame.from_dict(annotator_dict_cpt)
    #
    # annotator_sent_sentiment_df.to_csv('data/annotator_sent_sentiment.csv')
    # annotator_sent_aspect_df.to_csv('data/annotator_sent_aspect.csv')
    # annotator_cpt_df.to_csv('data/annotator_cpt_df.csv')
    #########################################################################################
    annotator_sent_sentiment_df = pd.read_csv('data/annotator_sent_sentiment.csv', index_col=0)
    annotator_sent_aspect_df = pd.read_csv('data/annotator_sent_aspect.csv', index_col=0)
    annotator_cpt_df = pd.read_csv('data/annotator_cpt_df.csv', index_col=0)

    print('calculate cpt df cohen kappa')
    get_cohen_kappa(annotator_cpt_df)  # 0.6324906687083851

    print('calculate sent sentiment df cohen kappa')
    get_cohen_kappa(annotator_sent_sentiment_df)  # 0.3086115047194282

    print('calculate sent aspect df cohen kappa')
    get_cohen_kappa(annotator_sent_aspect_df)  # 0.27315109925744585

    annotated_cpts_df = pd.read_csv('preprocessed_data/annotated_cpts_df.csv', index_col=0)  # 0.6324345513344987

    fless_annotated_cpts_df = pd.read_csv('preprocessed_data/annotated_cpts_df_fleiss.csv', index_col=0)
    get_fleiss_kappa(fless_annotated_cpts_df, True)

    annotated_sents_df = pd.read_csv('preprocessed_data/annotated_sentences_df.csv', index_col=0)
    get_fleiss_kappa(annotated_sents_df, False)

    human_performance_cpt(annotator_cpt_df)

    human_performance_sent(annotator_sent_aspect_df, 'aspect')
    human_performance_sent(annotator_sent_sentiment_df, 'sentiment')

    sent_agreement(annotated_sents_df)
