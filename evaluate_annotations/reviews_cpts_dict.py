import os
import json
from collections import defaultdict

import rootpath
import pandas as pd
import numpy as np


def load_reviews(file='data_processed/reviews.json'):
    """
    Load reviews from a json file.
    :param file:
    :return: a list of annotations
    """
    with open(file) as file:
        reviews = json.load(file)
    return reviews

def load_annotations(file='data_processed/annotations.json'):
    """
        Load annotations from a json file.
        :param file:
        :return: a list of annotations
        """
    with open(file) as file:
        annotations = json.load(file)
    return annotations

def get_cpt_dict(reviews):
    cpt_dict ={}  #
    for review in reviews:
        fileId = review['id']
        # category = '_'.join(review['categories'])
        review_list = review['review_list']

        cpt_ids = []
        for r in review_list:
            cpt = r['cpt']
            cpt_ids.append(int(cpt))
        cpt_ids_dedup= sorted(list(set(cpt_ids)))[:-1]

        cpt_dict[fileId]= cpt_ids_dedup
    return cpt_dict


#### correct the cpt ids in annotated_cpts_df.
### use reviews_cpts_df to generate change points in sent_df to get gold labels.
def load_cpts_df(cpt_dict, file="preprocessed_data/annotated_cpts_df.csv"):
    cpts_df = pd.read_csv(file, index_col=0)
    cpts_df_list=[]
    for fileId, group in cpts_df.groupby('fileId'):
        group['cpt_id'] = cpt_dict[fileId]
        cpts_df_list.append(group)

    cpts_df_new = pd.concat(cpts_df_list)
    new_indices = list(cpts_df_new['fileId'].astype(str).str.cat(cpts_df_new['cpt_id'].astype(str).str, sep='_'))
    cpts_df_new.index = new_indices
    cpts_df_new.drop(['iId'], axis=1, inplace=True)
    cpts_df_new.rename(columns={'cpt_id':'iId'}, inplace=True)
    return cpts_df_new





if __name__ == '__main__':
    reviews = load_reviews()
    cpt_dict = get_cpt_dict(reviews)
    cpts_df = load_cpts_df(cpt_dict)
    print(len(cpts_df))
    print(cpts_df.head())
    cpts_df.to_csv('preprocessed_data/reviews_cpts_df.csv')