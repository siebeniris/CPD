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
from joblib import Parallel, delayed

from keywords_dict import keywords
from utils import *
matplotlib.use('Agg')


def get_cpd_df(df, topic):
    df_reno = df[df[topic] == True]

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
    return df_cpd


def rolling_mean_diff(
    aspect_file: str, rolling_file: str, bkps_file: str, rpt_file: str
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
    df = pd.read_csv(aspect_file)
    cpd_df = get_cpd_df(df, "pool")

    win1 = int(round(len(cpd_df) / 15, 0))
    win2 = int(round(len(cpd_df) / 5, 0))

    rolling_mean = cpd_df.score.rolling(window=win1).mean()
    rolling_mean2 = cpd_df.score.rolling(window=win2).mean()
    # get diff
    diff = rolling_mean - rolling_mean2
    plt.figure(figsize=(40, 20))

    plt.plot(cpd_df.date, cpd_df.score, label="pool")
    plt.plot(cpd_df.date, rolling_mean, label='window ' + str(win1), color='orange')
    plt.plot(cpd_df.date, rolling_mean2, label='window ' + str(win2), color='magenta')
    plt.plot(cpd_df.date, diff, label='diff', color='green')
    plt.legend(loc='lower right')
    plt.savefig(rolling_file)
    plt.cla()
    plt.close()

    y = np.array(diff)
    # dates to deploy
    X = cpd_df.date[~np.isnan(y)].tolist()
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


if __name__ == '__main__':
    timer = Timer()
    data_dir = os.path.join(rootpath.detect(), 'data_backup2')
    annotation_dir = os.path.join(data_dir, 'annotation_by_topics', 'dataframes')
    sentiment_result_dir = os.path.join(data_dir, 'sentiment_analysis', 'results')

    timer.start()
    count=0
    for filename in os.listdir(annotation_dir):
        if count <3:
            output_dir = os.path.join(data_dir, 'annotation_by_topics', 'pool')
            rolling_file = os.path.join(output_dir, filename+'_window.png')
            bkp_file = os.path.join(output_dir, filename+'_bkp.json')
            rpt_file = os.path.join(output_dir, filename+'_rpt.png')

            aspect_file = os.path.join(annotation_dir, filename)
            rolling_mean_diff(aspect_file, rolling_file, bkp_file, rpt_file)
            count+=1


    timer.stop()