import os
import csv
import json
from typing import Any, List, Dict, Union
from datetime import date

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ruptures as rpt
import rootpath
import swifter

from window_rolling import *
from keywords import renovations, aspects

matplotlib.use('Agg')


def get_cpd_df(inputfile: str, outputfile: str) -> pd.DataFrame:
    """
    Get the change point detection dataframe
    :param inputfile: file from sentiment analysis result (has sentiment for each row)
    :param outputfile: file for writing out a new dataframe with polarity and topics.
    :return: change point detection dataframe for drawing the change point graphics.
    """
    if not os.path.exists(outputfile):
        df = pd.read_csv(inputfile)
        print('read file', inputfile)
        # change the date only to year-month-day
        df['date'] = df['date'].swifter.apply(lambda x: pd.to_datetime(x, errors='coerce')).dropna().dt.strftime('%Y-%m-%d')
        print(df['date'].to_list()[:10])
        df['sentence'] = df['sentence'].astype(str)
        df['sentiment'] = df['sentiment'].swifter.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
        df['lemma'] = df['lemma'].astype(str)
        # given the sentiment to output the polarity
        df.loc[df['sentiment'] >= 0.3, 'polarity'] = 1
        df.loc[df['sentiment'] <= 0, 'polarity'] = -1
        df.loc[(df['sentiment'] > 0) & (df['sentiment'] < 0.3), 'polarity'] = 0
        print(df['polarity'].to_list()[:10])

        # lower the lemma and split() into list and find the common words with keyword list.
        df['renovation'] = df.swifter.apply(lambda x: sum([y in x.lemma.lower().split() for y in renovations]) > 0, axis=1)
        df['other'] = df.swifter.apply(lambda x: sum([y in x.lemma.lower().split() for y in aspects]) > 0, axis=1)
        print(df['renovation'].to_list()[:10])
        print(df['other'].to_list()[:10])
        df = df.sort_values('date')

        df.to_csv(outputfile)
        print('Write dataframe with polarity and topics to ', outputfile)
    else:
        df = pd.read_csv(outputfile)
        print('File exists, load dataframe from ', inputfile)

    df_reno = df[df['renovation'] == True]

    # aggreagte the dataframe by the date and its polarity scores for each date.
    dates = []
    scores = []
    # sort dates here.
    for date in sorted(list(set(df_reno.date.to_list()))):
        score = 0

        rows = df_reno[df_reno['date'] == date]
        for l in range(len(rows)):
            row = rows.iloc[l]
            score += row.polarity
        scores.append(score / len(rows))
        dates.append(date)

    # get the DataFrame by dates and scores
    df_cpd = pd.DataFrame(list(zip(dates, scores)), columns=['date', 'score'])
    print('The length of the change point dataframe: ', len(df_cpd))
    # df other topics.
    return df_cpd


def rolling_mean_diff(
        inputfile: str, aspect_file: str, rolling_file: str, bkps_file: str, rpt_file: str
) -> Dict[str, Union[Union[list, List[int]], Any]]:
    """
    Draw the mean difference from short/long rolling window mean .
    :param inputfile: result Dataframe from sentiment analysis
    :param aspect_file: aspect dataframe from get_cpd_df
    :param rolling_file: output file for the graph.
    :param bkps_file: change points output from rupture dynamic programming method.
    :param rpt_file: outputfile of change point detection graph from dynamic programming method.
    :return: the dictionary of (list of dates, list of scores, list of change points).
    """
    cpd_df = get_cpd_df(inputfile, aspect_file)

    win1 = int(round(len(cpd_df) / 15, 0))
    win2 = int(round(len(cpd_df) / 5, 0))

    rolling_mean = cpd_df.score.rolling(window=win1).mean()
    rolling_mean2 = cpd_df.score.rolling(window=win2).mean()
    # get diff
    diff = rolling_mean - rolling_mean2
    plt.figure(figsize=(40, 20))

    plt.plot(cpd_df.date, cpd_df.score, label='renovation')
    plt.plot(cpd_df.date, rolling_mean, label='window ' + str(win1), color='orange')
    plt.plot(cpd_df.date, rolling_mean2, label='window ' + str(win2), color='magenta')
    plt.plot(cpd_df.date, diff, label='diff', color='green')
    plt.legend(loc='lower right')
    plt.savefig(rolling_file)
    plt.cla()
    plt.close()

    y = np.array(diff)
    # dates to deploy
    X =cpd_df.date[~np.isnan(y)].tolist()
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

        return d


def get_point_for_df(inputfile, bkps, dates, scores):
    """
    Get change points for producing relevant reviews.
    :param inputfile: input file for dataframe with polarity and aspects.
    :param bkps:
    :param dates:
    :param scores:
    :return:
    """
    # get df from aspects dir.
    df = pd.read_csv(inputfile)
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

    # if rows are between this period, get point.
    # df.loc[(df['date'] <= dates[period_ids[0][1]]), 'point'] = 0
    # get period scores
    period_scores = {}
    # period_scores[(0, bkps[0])] = np.mean(scores[:bkps[0]])
    for idx, period in period_ids.items():
        start, end = period
        period_scores[(start, end)] = np.mean(scores[start:end])
        df.loc[(df['date'] > dates[start]) & (df['date'] <= dates[end - 1]), 'point'] = idx
    print('scores between period=============>')
    print(period_scores)
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

    for idx, score_pair in enumerate(score_pairs):
        one, two = score_pair
        z = two - one
        if idx == 0:  # starting point.
            if z > 0:
                trends.append(('low', bool(one > 0)))
                trends.append(('high', bool(two > 0)))
            else:
                trends.append(('high', bool(one > 0)))
                trends.append(('low', bool(two > 0)))
        else:
            if z < 0:
                trends.append(('low', bool(one > 0)))
            else:
                trends.append(('high', bool(two > 0)))

    assert len(period_scores) == len(trends)
    return df, trends


def get_dataset(df, trends, neg_fac=10):
    """
    Get Dataset
    :param df: from aspects dir.
    :param trends:
    :param neg_fac: negative factor.
    :return:
    """
    reno_df = df[(df['renovation'] == True) & (df['other'] == True)]
    other_df = df[(df['renovation'] == False) & (df['other'] == True)]

    # uids = {}
    sentences = dict()
    print('trends :', trends)

    for idx in range(len(trends)):
        trend, _ = trends[idx]
        if trend == 'low':
            df_idx = reno_df[(reno_df['point'] == idx) & (reno_df['polarity'] == -1)]
            df_other_idx = other_df[(other_df['point'] == idx) & (other_df['polarity'] == -1)]
            if len(df_idx) > neg_fac:
                df_idx = df_idx.sample(neg_fac)
            if len(df_other_idx) > neg_fac:
                df_other_idx = df_other_idx.sample(neg_fac)

        if trend == 'high' :
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
        # print(sent_date)
        # uid_date = sorted(list(zip(uidss, dates)), key=lambda x: x[1])
        # print(trends[idx + 1])
        #
        # next_trend, _ = trends[idx + 1]
        # print(idx, next_trend)
        sentences[idx] = sent_date

        # uids[(idx, trend)] = uid_date

    return sentences, trends


def write_out(sentences, trends,  outputfile):
    with open(outputfile, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['date', 'sentence', 'typical?(renovation) yes or no'])
        for idx, reviews in sentences.items():
            for review in reviews:
                sentence, date = review
                csvwriter.writerow([date, sentence, ''])
            if idx < len(trends)-1:
                next_trend = trends[idx + 1][0]
                print('writing ', next_trend)
                if next_trend == 'low':
                    csvwriter.writerow(['change point: ', 'going down', ''])
                if next_trend == 'high':
                    csvwriter.writerow(['change point: ', 'going up', ''])



if __name__ == '__main__':
    data_dir = os.path.join(rootpath.detect(), 'data')
    annotation_dir = os.path.join(data_dir, 'annotation_samples', 'annotations')
    sentiment_result_dir = os.path.join(data_dir, 'sentiment_analysis', 'results')

    # for diff graphics
    diff_window_dir = os.path.join(data_dir, 'annotation_samples', 'diff')
    bkps_dir = os.path.join(data_dir, 'annotation_samples', 'bkps')
    rpts_dir = os.path.join(data_dir, 'annotation_samples', 'rpts')
    aspects_df_dir = os.path.join(data_dir, 'annotation_samples', 'aspects')

    neg_fac = 10
    count = 0
    for filename in os.listdir(sentiment_result_dir):
        if count < 4:
            print('filename ', filename)
            try:
                filepath = os.path.join(sentiment_result_dir, filename)
                diff_file = os.path.join(diff_window_dir, filename + '.png')
                bkp_file = os.path.join(bkps_dir, filename + '.json')
                rpt_file = os.path.join(rpts_dir, filename + '.png')
                aspect_file = os.path.join(aspects_df_dir, filename)
                outputfile = os.path.join(annotation_dir, filename)

                # if not anntation file.
                if not os.path.exists(outputfile):
                    # if there is sentiment analysis result.
                    if os.path.isfile(filepath):
                        # check if there is bkp_file.
                        if os.path.exists(bkp_file):
                            print('processing bkps file')
                            with open(bkp_file) as file:
                                bkps_dict = json.load(file)
                            bkps = bkps_dict['bkps']
                            scores = bkps_dict['scores']
                            dates = bkps_dict['dates']
                            assert len(bkps) == 4
                        else:
                            bkps_dict = rolling_mean_diff(filepath, aspect_file, diff_file, bkp_file, rpt_file)
                            bkps = bkps_dict['bkps']
                            scores = bkps_dict['scores']
                            dates = bkps_dict['dates']
                        print('date samples', dates[:4])
                        print('producing ...', filename)
                        if os.path.exists(aspect_file):
                            df_new, trends = get_point_for_df(aspect_file, bkps, dates, scores)
                            if len(trends) > 0:
                                sentences, trends = get_dataset(df_new, trends, neg_fac=neg_fac)

                                write_out(sentences,trends, outputfile)
                                print('finished ', filename)
                                print('=====' * 30)
                            else:
                                print(trends, 'empty...')
                else:
                    print(outputfile, ' exists...')

            except Exception:
                print(filename, ' problem..')

            print('=====' * 40)
            count += 1
