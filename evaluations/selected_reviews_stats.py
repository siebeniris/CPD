import json
from collections import defaultdict

import pandas as pd
import numpy as np


def load_reviews(file):
    """
    Load reviews from file.
    :param file: filepath
    :return: reviews from database.
    """
    with open(file) as reader:
        data = json.load(reader)
    return data


def load_annotations(file):
    """
    Load annotations from file.
    :param file: filepath
    :return: annotations from database.
    """
    with open(file) as reader:
        data = json.load(reader)
    return data['annotations']


def get_stats_annotations(annotations, cpt_dict):
    # cpt_dict : fileId -> {'nr_of_cpts': nr_cpts}

    for annotation in annotations:
        fileId = int(annotation['fileId'])
        nr_cpts = cpt_dict[fileId]['nr_of_cpts']
        if annotation['cptAnswer']:
            anno_nr_cpts = len(annotation['cptAnswer'])
            if nr_cpts != anno_nr_cpts:
                print(fileId, nr_cpts, anno_nr_cpts)



def get_stats_reviews(reviews):
    """
    Get some stats.
    :param reviews:
    :return:
    """
    sent_df_dict = defaultdict(dict)
    cpt_df_dict = defaultdict(dict)
    cpt_dict = defaultdict(dict)

    sent_stats_dict = defaultdict(int)
    cpt_stats_dict = defaultdict(int)
    categories_dict = defaultdict(int)

    for review in reviews:
        idx = int(review['id'])
        category = '_'.join(review['categories'])
        categories_dict[category] += 1
        review_list = review['review_list']
        name = review['name'].split('_')[-1]

        cpts = []
        for r in review_list:
            cpt = int(r['cpt'])
            cpts.append(cpt)

            r_id = r['id']
            id_ = '_'.join([str(idx), str(r_id)])
            sent_df_dict[id_] = {
                "sent_id": r_id,
                "file_id": idx,
                "cpt": cpt,
                "category": category,
                'hotel_id': name

            }

        nr_cpts = len(list(set(cpts))) - 1

        cpt_dict[idx] = {
            "nr_of_sentences": len(review_list),
            "nr_of_cpts": nr_cpts
        }
        dedup_sorted_cpts = sorted(list(set(cpts)))
        if len(dedup_sorted_cpts) > 1:
            for idxx, cpt in enumerate(dedup_sorted_cpts[:-1]):
                id_ = str(idx) + '_' + str(idxx)
                cpt_df_dict[id_] = {
                    "file_id": idx,
                    "cpt":cpt,
                    "category":category,
                    "hotel_id":name
                }

        cpt_stats_dict[nr_cpts] += 1
        sent_stats_dict[len(review_list)] += 1

    sent_sum = sum([k * v for k, v in sent_stats_dict.items()])
    print("In total there are {} sentences".format(sent_sum))

    print("====change points ====")
    print(cpt_stats_dict)
    cpts_sum = sum([k * v for k, v in cpt_stats_dict.items()])
    print("In total there are {} change points".format(cpts_sum))
    print("categories:", categories_dict)

    # file_sent_id, sent_id, file_id, cpt
    df = pd.DataFrame.from_dict(sent_df_dict, orient="index")
    df = df.sort_values(by=['file_id', 'sent_id'])
    df.to_csv("stats_reviews.csv")

    cpt_df = pd.DataFrame.from_dict(cpt_df_dict, orient="index")
    cpt_df = cpt_df.sort_values(by=['file_id', 'cpt'])
    cpt_df.to_csv("stats_cpts.csv")
    return df, cpt_dict


if __name__ == '__main__':
    db_path = 'data_processed/reviews.json'

    reviews = load_reviews(db_path)
    df, cpt_dict = get_stats_reviews(reviews)

    # annotations = load_annotations(db_path)
    # get_stats_annotations(annotations, cpt_dict)

    # {1: 252, 5: 3, 3: 36, 2: 90, 0: 32, 4: 6, 6: 1}
    # {'facility': 69, 'pool': 66, 'renovation_room': 62, 'restaurant': 72,
    # 'room': 86, 'reception': 65}
