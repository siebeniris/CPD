import os
import csv
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

from window_rolling import *

other_topics_keywords = ['suite', 'minibar', 'bathroom', 'view', 'shower', 'wifi', 'balcony', 'porch',
                         'tv', 'furniture', 'bathtub',
                         'restaurant', 'menu', 'dinner', 'breakfast', 'lunch', 'bar', 'café', 'dining', 'dish',
                         'snack', 'dessert',
                         'wellness', 'pool', 'fitness', 'amenity', 'facility', 'atmosphere', 'cost', 'vibe',
                         'beach', 'sea', 'ocean', 'lake', 'casino', 'shuttle', 'sport',
                         'transport', 'maintenance',
                         'lobby', 'parking', 'park', 'golf', 'hotel',
                         'reception', 'service', 'staff', 'concierge',
                         'location']


def rolling_mean_diff(inputfile, bkps_file, rpt_file):
    """
    Load DataFrame from renovation
    :param inputfile:
    :return:
    """
    df = pd.read_csv(inputfile)
    cpd_df = get_cpd_df(df)

    win1 = int(round(len(cpd_df) / 15, 0))
    win2 = int(round(len(cpd_df) / 5, 0))

    rolling_mean = cpd_df.score.rolling(window=win1).mean()
    rolling_mean2 = cpd_df.score.rolling(window=win2).mean()

    # get diff
    diff = rolling_mean - rolling_mean2
    y = np.array(diff)
    # dates to deploy
    X = np.array(cpd_df.date)[~np.isnan(y)].tolist()
    y = y[~np.isnan(y)]
    scores = y

    if win2 > 10 and isinstance(X, list):
        plt.rcParams.update({'figure.max_open_warning': 0})
        model = "normal"
        algo = rpt.Dynp(model=model, min_size=3, jump=5).fit(scores)
        my_bkps = algo.predict(n_bkps=3)
        rpt.display(scores, my_bkps, figsize=(10, 6))
        plt.title('Change Point Detection: Dynamic Programming Search Method')
        plt.savefig(rpt_file)
        plt.cla()
        plt.close()

        bkps = [int(x) for x in my_bkps]
        d = {
            'dates': X,
            'scores': scores.tolist(),
            'bkps': bkps
        }
        with open(bkps_file, 'w') as file:
            json.dump(d, file)

        return df, d


def get_point_for_df(df, bkps, dates, scores):
    ## get the period dictionary
    periods = list(zip(bkps[:-1], bkps[1:]))
    period_ids = {}
    period_ids[0] = (0, bkps[0])
    count = 1
    for z in periods:
        x, y = z
        period_ids[count] = (x, y)
        count += 1
    # {0: (0, 85), 1: (85, 155), 2: (155, 475), 3: (475, 647)}

    # reno_df
    df.loc[(df['date'] <= dates[period_ids[0][1]]), 'point'] = 0
    # get period scores
    period_scores = {}
    period_scores[(0, bkps[0])] = np.mean(scores[:bkps[0]])
    for idx, period in period_ids.items():
        if idx >= 1:
            start, end = period
            period_scores[(start, end)] = np.mean(scores[start:end])
            df.loc[(df['date'] > dates[start]) & (df['date'] <= dates[end - 1]), 'point'] = idx
    # period_scores
    # {(0, 85): 0.07696766307166987,
    #  (85, 155): -0.12219812240516177,
    #  (155, 475): 0.014581536116862245,
    #  (475, 647): -0.14434456936352785}

    scores_list = list(period_scores.values())
    print('scores list --> ', scores_list)
    score_pairs = list(zip(scores_list[:-1], scores_list[1:]))
    # [(0.07696766307166987, -0.12219812240516177),
    #  (-0.12219812240516177, 0.014581536116862245),
    #  (0.014581536116862245, -0.14434456936352785)]
    # to compute the trends.

    # ['high', 'low', 'high', 'low']
    trends = []

    start_score = score_pairs[0][0]
    for x1, y1 in score_pairs:
        z = y1 - x1
        if start_score < 0:
            if z > 0:
                trends.append('low')
                trends.append('high')

        if start_score > 0:
            if z < 0:
                trends.append('high')
                trends.append('low')

    # df other topics.
    df['other'] = df.apply(lambda x: sum([y in x.lemma for y in other_topics_keywords]) > 0, axis=1)

    return len(period_ids), df, trends


def get_dataset(LEN, df, trends, neg_fac=10):
    """
    Get Dataset
    :param LEN:
    :param df:
    :param trends:
    :param neg_fac: negative factor.
    :return:
    """
    reno_df = df[(df['renovation'] == True) & (df['other'] == True)]
    other_df = df[(df['renovation'] == False) & (df['other'] == True)]

    # uids = {}
    sentences = {}
    if len(trends)>0:
        for idx in range(LEN):
            trend = trends[idx]
            print(trend)
            if trend == 'low':
                df_idx = reno_df[(reno_df['point'] == idx) & (reno_df['polarity'] == -1)]
                df_other_idx = other_df[(other_df['point'] == idx) & (other_df['polarity'] == -1)]
                if len(df_idx) > neg_fac:
                    df_idx = df_idx.sample(neg_fac)
                if len(df_other_idx) > neg_fac:
                    df_other_idx = df_other_idx.sample(neg_fac)

            if trend == 'high':
                df_idx = reno_df[(reno_df['point'] == idx) & (reno_df['polarity'] == 1)]
                df_other_idx = other_df[(other_df['point'] == idx) & (other_df['polarity'] == 1)]
                if len(df_idx) > neg_fac:
                    df_idx = df_idx.sample(neg_fac)
                if len(df_other_idx) > neg_fac:
                    df_other_idx = df_other_idx.sample(neg_fac)

            sent = df_idx.sentence.to_list() + df_other_idx.sentence.to_list()
            dates = df_idx.date.to_list() + df_other_idx.date.to_list()
            # uidss = df_idx.uid.to_list() + df_other_idx.uid.to_list()

            sent_date = sorted(list(zip(sent, dates)), key=lambda x: x[1])
            # uid_date = sorted(list(zip(uidss, dates)), key=lambda x: x[1])
            sentences[(idx, trend)] = sent_date
            # uids[(idx, trend)] = uid_date

    return sentences


def write_out(sentences, outputfile):
    with open(outputfile, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['date', 'sentence', 'typical?(renovation) yes or no'])
        for idx_trend, reviews in sentences.items():
            idx, trend = idx_trend
            for review in reviews:
                sentence, date = review
                csvwriter.writerow([date, sentence, ''])
            if trend == 'low':
                csvwriter.writerow(['change point: ', 'going up', ''])
            if trend == 'high':
                csvwriter.writerow(['change point: ', 'going down', ''])


if __name__ == '__main__':
    data_dir = "/home/yiyi/Documents/masterthesis/CPD/data_backup2"
    annotation_dir = os.path.join(data_dir, 'annotation_samples', 'annotations')
    renovation_dir = os.path.join(data_dir, 'rolling_window', 'renovation')

    # for diff graphics
    bkps_dir = os.path.join(data_dir, 'annotation_samples', 'bkps')
    rpts_dir = os.path.join(data_dir, 'annotation_samples', 'rpts')

    neg_fac = 10
    for filename in os.listdir(renovation_dir):
        try:
            filepath = os.path.join(renovation_dir, filename)
            bkp_file = os.path.join(bkps_dir, filename + '.json')
            rpt_file = os.path.join(rpts_dir, filename + '.png')
            outputfile = os.path.join(annotation_dir, filename + '.csv')
            if not os.path.exists(outputfile):
                if os.path.isfile(filepath):
                    if os.path.exists(bkp_file):
                        print('processing bkps file')
                        with open(bkp_file) as file:
                            bkps_dict = json.load(file)
                        bkps = bkps_dict['bkps']
                        scores = bkps_dict['scores']
                        dates = bkps_dict['dates']
                        df = pd.read_csv(filepath)
                        assert len(bkps) == 4
                    else:
                        df, bkps_dict = rolling_mean_diff(filepath, bkp_file, rpt_file)
                        bkps = bkps_dict['bkps']
                        scores = bkps_dict['scores']
                        dates = bkps_dict['dates']

                    print('producing ...', filename)
                    LEN, df_new, trends = get_point_for_df(df, bkps, dates, scores)
                    if len(trends)>0:
                        sentences = get_dataset(LEN, df_new, trends, neg_fac=neg_fac)
                        write_out(sentences, outputfile)
                        print('finished ', filename)
                        print('====='*30)
                    else:
                        print(trends, 'empty...')
            else:
                print(outputfile, ' exists...')

        except Exception:
            print(filename, ' problem..')
