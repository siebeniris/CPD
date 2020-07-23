import os
import json
from collections import defaultdict

import rootpath
import pandas as pd
import numpy as np


def load_annotations(file='data_processed/annotations.json'):
    """
    Load annotations from a json file.
    :param file:
    :return: a list of annotations
    """
    with open(file) as file:
        data = json.load(file)
    return data



def get_annotations_dict(annotations):
    """
    Get annotations_dict.
    :param annotations: loaded from database
    :return: dictionary of annotations.
    """
    new_annotations = defaultdict(list)
    for a in annotations:
        idx = int(a['fileId'])
        new_annotations[idx].append(a)

    annotations_dict = defaultdict(list)
    for idx, al in new_annotations.items():
        adict = defaultdict(list)

        for a in al:
            username = a['username']
            adict[username].append(a)
        # for fleiss kappa calcuation of cpt
        for username, a in adict.items():
            if len(a) > 1:
                sorted_a = sorted(a, key=lambda i: i['date'])
                # get the most recent annotation from one user.
                annotations_dict[idx].append(sorted_a[-1])
            else:

                annotations_dict[idx].append(a[0])
    return annotations_dict


def df_dict(annotations_dict):
    """
    To prepare dictionary for DataFrame from the annotations dictionary.
    :param annotations_dict:
    :return: dataframe dictioanroes.
    """
    # idx, sentID, no_aspect, no_sentiment, yes_two
    dataframe_dict = dict()
    cpts_dict = dict()
    no_cpts = 0
    for idx, alist in annotations_dict.items():

        for a in alist:
            annotation = a['annotation']
            username = a['username']
            # for fleiss kappa caculation
            if username!='yiyi':
                if a['cptAnswer']:
                    cpts = a['cptAnswer']

                    for nr, cpt in cpts.items():
                        id_nr = str(idx) + '_' + nr

                        if id_nr not in cpts_dict:
                            cpts_dict[id_nr] = defaultdict(int)
                        cpts_dict[id_nr][cpt] += 1
            else:
                no_cpts += 1
            if username != 'yiyi':
                for anno in annotation:
                    id_ = anno['id']

                    yes_aspect_ = anno.get('yes_aspect', False)
                    yes_sentiment_ = anno.get('yes_sentiment', False)

                    id_sentid = str(idx) + '_' + str(id_)

                    if id_sentid not in dataframe_dict:
                        dataframe_dict[id_sentid] = {
                            'two-yes': 0,
                            'two-no': 0,
                            'ya-ns': 0,
                            'ys-na': 0
                        }

                    if not yes_aspect_ and not yes_sentiment_:
                        dataframe_dict[id_sentid]['two-no'] += 1
                    if not yes_sentiment_ and yes_aspect_:
                        dataframe_dict[id_sentid]['ya-ns'] += 1
                    if yes_sentiment_ and not yes_aspect_:
                        dataframe_dict[id_sentid]['ys-na'] += 1
                    if yes_sentiment_ and yes_aspect_:
                        dataframe_dict[id_sentid]['two-yes'] += 1

    print('no cpts:', no_cpts)

    return dataframe_dict, cpts_dict


def get_df_from_dict(df_dict):
    """
    Get DataFrame from a df dictionary
    :param df_dict:
    :return: processed df.
    """
    df = pd.DataFrame.from_dict(df_dict, orient='index')
    df = df.fillna(0)  # fillna with 0.
    # convert the float to int.
    cols = df.columns
    df[cols] = df[cols].applymap(np.int64)
    # total from categories for each unit.

    total_col = df.sum(axis=1)
    df['Total'] = total_col

    print('totals: ', list(set(total_col)))

    indices = df.index  # 0_0, 0_1, 0_2....
    fileId, ind_ids = [], []
    for id_ in indices:
        fileid, ind_id = id_.split('_')
        fileId.append(int(fileid))
        ind_ids.append(int(ind_id))

    df['fileId'] = fileId
    df['iId'] = ind_ids

    sorted_df = df.sort_values(by=['fileId', 'iId'])
    return sorted_df


def get_unfinished_df(sorted_df, cpt=False):
    """
    Get unfinished dataframe.
    :param sorted_df:
    :return:
    """

    if cpt:
        df = sorted_df[(sorted_df['Total'] == 2) | (sorted_df['Total'] == 3)]
        c_df = df[(df['c'] == 2) | (df['c'] == 3)]
        print('no_change:', len(c_df))
        print('change points:', len(df) - len(c_df))
        print('total unit:', len(df))
        print('total finished:', len(list(set(df['fileId']))))
        unfinished_df = df.loc[(df['a'] == 1) | (df['b'] == 1) | (df['c'] == 1)]
    else:
        # df = sorted_df[(sorted_df['Total'] == 2) | (sorted_df['Total'] == 4)]
        df = sorted_df
        print('total unit:', len(df))
        print('total finished:', len(list(set(df['fileId']))))
        unfinished_df = df.loc[(df['ys-na'] == 1) | (df['ya-ns'] == 1) | (df['two-yes'] == 1) |
                               (df['two-no'] == 1)]

    print('disagreed annotated units:', len(unfinished_df))

    dedup_files = len(list(set(unfinished_df['fileId'])))
    print('unfinished dataframe files:', dedup_files)
    return unfinished_df, df





if __name__ == '__main__':

    annotations = load_annotations()
    annotations_dict = get_annotations_dict(annotations)
    sentences_dict, cpts_dict = df_dict(annotations_dict)

    cpts_df = get_df_from_dict(cpts_dict)
    cpts_df.to_csv('preprocessed_data/annotated_cpts_df_fleiss.csv')
    # sentences_df = get_df_from_dict(sentences_dict)
    # sentences_df.to_csv('annotated_sentences_df_recat.csv')
    #
    #
    # unfinish_sent_df, _ = get_unfinished_df(sentences_df)
    # unfinish_cpt_df, _ = get_unfinished_df(cpts_df, cpt=True)

    # unfinish_cpt_df.to_csv('unfinished_cpt_df.csv')
    # unfinish_sent_df.to_csv('unfinished_sentence_df.csv')
