import os
import swifter
import json
import csv
import rootpath

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from topics import topics as topic_dict

wbs = importr("wbs")


def get_cpd_df(file, aspects):
    df = pd.read_csv(file)
    # df['date'] = df['date'].astype('datetime64')
    # df['sentiment'] = df['sentiment'].swifter.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
    # df['lemma'] = [json.loads(x) for x in df['lemma']]
    # df['LEN'] = [len(x) for x in df['lemma']]
    # df = df[df['LEN'] >= 3]
    # for keyword, values in topic_dict.items():
    #     df[keyword] = df.swifter.apply(lambda x: sum([y in x.lemma.lower().split() for y in values]) > 0,
    #                                    axis=1)
    df.drop_duplicates(['sentence'], keep='first', inplace=True)  # drop duplicates based on sentence
    df = df.sort_values('date')

    assert aspects == ['room', 'restaurant', 'facility', 'pool', 'family', 'price', 'atmosphere', 'fitness',
                       'transport', 'beach', 'entertainment', 'venue', 'parking', 'reception', 'renovation']


    conditional = (df['room']==True) | (df['restaurant']==True) | (df['facility']==True) | (df['pool']==True) \
                  | (df['family']==True) | (df['price']==True) | (df['atmosphere']==True) | (df['fitness']==True)\
                  | (df['transport']==True) | (df['beach']==True) | (df['entertainment']==True)\
                  | (df['venue']==True) | (df['parking']==True) | (df['reception']==True) | (df['renovation']==True)
    df_new = df[conditional]

    print('length of df :', len(df))
    print('length of df_new', len(df_new))
    cpd_df = df_new.groupby(['date', 'uid'])['sentiment'].mean().reset_index()
    cpd_df = cpd_df.groupby('date')['sentiment'].mean().reset_index()

    # print("lenght of cpd_df:", len(cpd_df))
    return cpd_df, df


def emas(cpd_df, emas_png):
    # to validate the trend shown in wbs output
    # ema
    plt.rcParams.update({'figure.max_open_warning': 0})
    cpd_df.set_index('date', inplace=True)
    cpd_df['EMA_100'] = cpd_df['sentiment'].ewm(span=100, adjust=False).mean()
    cpd_df['EMA_50'] = cpd_df['sentiment'].ewm(span=50, adjust=False).mean()
    cpd_df['EMA_10'] = cpd_df['sentiment'].ewm(span=10, adjust=False).mean()
    plt.figure(figsize=[20, 10])
    plt.grid(True)
    plt.plot(cpd_df['sentiment'], label='score')
    plt.plot(cpd_df['EMA_10'], label='EMA-10')
    plt.plot(cpd_df['EMA_50'], label='EMA-20')
    plt.plot(cpd_df['EMA_100'], label='EMA-100')
    plt.legend(loc=2)
    plt.savefig(emas_png)
    plt.cla()
    plt.close()


def wild_binary_segmentation(cpd_df):
    """
    Applying wild binary segmentation change-point-detection on sentiment scores.
    :param cpd_df:
    :return:
    """
    try:
        sentiments = cpd_df.sentiment.to_list()
        sentiments_r = ro.FloatVector(sentiments)
        w = wbs.wbs(sentiments_r)
        w_cpt = wbs.changepoints(w)
        cpt = w_cpt.rx2("cpt.ic").rx2("ssic.penalty")

        cpt = list(cpt)
        return cpt
    except Exception as mg:
        print(mg)


def get_info_list(df):
    df = df.sort_values('date')
    dates = df.date.to_list()
    uids = df.uid.to_list()
    lemmas = df.lemma.to_list()
    sentences = df.sentence.to_list()
    sentiments = df.sentiment.to_list()
    return list(zip(dates, uids, lemmas, sentences, sentiments))


def select_reviews(cpt, cpd_df, df):
    # total length of dates
    LEN = len(cpd_df)
    df = df[df['LEN'] > 2]
    cpt = sorted([0] + [int(x) for x in cpt] + [LEN])
    # print("change point ==> ", cpt)
    cpd_df['date'] = cpd_df.index
    # get the date periods.
    cpt_periods = list(zip(cpt[:-1], cpt[1:]))

    dates_periods = [(cpd_df.iloc[x].date, cpd_df.iloc[y - 1].date)
                     for x, y in cpt_periods]
    sentiment = cpd_df.sentiment.to_list()

    sentiment_mean = [np.mean(sentiment[x:y]) for x, y in cpt_periods]
    # print(sentiment_mean)

    sentences = dict()
    # one change points:
    if len(cpt_periods) == 2:
        assert len(sentiment_mean) == 2
        trend = int(np.sign(sentiment_mean[-1] - sentiment_mean[0]))
        dates1, dates2 = dates_periods
        sent1, sent2 = sentiment_mean
        df_1 = df[(df['date'] > dates1[0]) & (df['date'] < dates1[1])]
        df_2 = df[(df['date'] > dates2[0]) & (df['date'] < dates2[1])]

        if trend == -1:
            df_1 = df_1[df_1['sentiment'] > sent1]
            if len(df_1) > 10:
                df_1 = df_1.sample(10)
            sentences[0] = get_info_list(df_1)

            df_2 = df_2[df_2['sentiment'] < sent2]
            if len(df_2) > 10:
                df_2 = df_2.sample(10)
            sentences[1] = get_info_list(df_2)

        if trend == 1:
            df_1 = df_1[df_1['sentiment'] < sent1]
            if len(df_1) > 10:
                df_1 = df_1.sample(10)
            sentences[0] = get_info_list(df_1)

            df_2 = df_2[df_2['sentiment'] > sent2]
            if len(df_2) > 10:
                df_2 = df_2.sample(10)
            sentences[1] = get_info_list(df_2)

    if len(cpt_periods) >= 3:
        # between 1,...,n-1
        sentiment_mean_periods = list(zip(sentiment_mean[1:], sentiment_mean[:-1]))[:-1]
        # print("sentiment mean periods :", sentiment_mean_periods)
        trends_numeric = list(zip(sentiment_mean[:-1], sentiment_mean[1:]))
        trends = [int(np.sign(y - x)) for x, y in trends_numeric]
        first_trend, last_trend = trends[0], trends[-1]
        first_date, last_date = dates_periods[0], dates_periods[-1]
        df_first = df[(df['date'] > first_date[0]) & (df['date'] < first_date[1])]
        df_last = df[(df['date'] > last_date[0]) & (df['date'] < last_date[1])]
        sent_first, sent_last = sentiment_mean[0], sentiment_mean[-1]

        # the first element.
        if first_trend == -1:
            df_first = df_first[df_first['sentiment'] > sent_first]
            if len(df_first) > 10:
                df_first = df_first.sample(10)
            sentences[0] = get_info_list(df_first)

        if first_trend == 1:
            df_first = df_first[df_first['sentiment'] < sent_first]
            if len(df_first) > 10:
                df_first = df_first.sample(10)
            sentences[0] = get_info_list(df_first)

        for idx, sentiment_mean in enumerate(sentiment_mean_periods):
            min_, max_ = min(sentiment_mean), max(sentiment_mean)
            date_start, date_end = dates_periods[idx + 1]
            df_ = df[(df['date'] > date_start) & (df['date'] < date_end)]
            df_ = df_[(df_['sentiment'] > min_) & (df_['sentiment'] < max_)]

            if len(df_) > 10:
                df_ = df_.sample(10)
            sentences[idx + 1] = get_info_list(df_)

        # the last element.
        last = len(cpt_periods) - 1
        if last_trend == -1:
            df_last = df_last[df_last['sentiment'] > sent_last]
            if len(df_last) > 10:
                df_last = df_last.sample(10)
            sentences[last] = get_info_list(df_last)

        if last_trend == 1:
            df_last = df_last[df_last['sentiment'] < sent_last]
            if len(df_last) > 10:
                df_last = df_last.sample(10)
            sentences[last] = get_info_list(df_last)

    return sentences


def write_out_to_json(sentences, outputfile):
    print('save to ', outputfile)
    with open(outputfile, 'w') as file:
        json.dump(sentences, file)


def write_out_to_csv(sentences, outputfile):
    print('save to ', outputfile)
    with open(outputfile, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['date', 'uid', 'lemma', 'sentence', 'sentiment', 'typical?(renovation) yes or no'])
        for idx, sentences in sentences.items():
            for el in sentences:
                csvwriter.writerow(list(el) + [''])
            csvwriter.writerow(["change point ", str(idx + 1)])


if __name__ == '__main__':
    root_dir = rootpath.detect()
    cpd_aspects = os.path.join(root_dir, 'data_backup2', 'cpd_aspects')

    output_dir = os.path.join(root_dir, "data_backup2", "select_reviews_0606")

    aspects = list(topic_dict.keys())

    for filename in os.listdir(cpd_aspects):
        filepath = os.path.join(cpd_aspects, filename)
        if os.path.isfile(filepath):
            print('load file ', filepath)

            # for aspect in aspects:
            # get all the aspects for a hotel.
            aspect_dir = os.path.join(output_dir, "general")

            if not os.path.exists(aspect_dir): os.mkdir(aspect_dir)

            reviews_dir = os.path.join(aspect_dir, "reviews")

            json_dir = os.path.join(reviews_dir, "json_file")
            csv_dir = os.path.join(reviews_dir, "csv_dir")
            emas_dir = os.path.join(aspect_dir, "emas")

            if not os.path.exists(reviews_dir): os.mkdir(reviews_dir)
            if not os.path.exists(json_dir): os.mkdir(json_dir)
            if not os.path.exists(csv_dir): os.mkdir(csv_dir)
            if not os.path.exists(emas_dir): os.mkdir(emas_dir)
            jsonfile = os.path.join(json_dir, filename + '.json')
            csvfile = os.path.join(csv_dir, filename + '.csv')
            exceptionfile = os.path.join(reviews_dir, filename)
            if not os.path.exists(exceptionfile):
                if not os.path.exists(jsonfile) or not os.path.exists(csvfile):
                    cpd_df, df = get_cpd_df(filepath, aspects)
                    # emas plot.
                    ems_file = os.path.join(emas_dir, filename + '.png')
                    if not os.path.exists(ems_file):
                        emas(cpd_df, os.path.join(emas_dir, filename + '.png'))

                    LEN_REVIEWS= len(cpd_df)
                    cpt = wild_binary_segmentation(cpd_df)
                    try:
                        print("change points detected :", cpt)
                        sentences = select_reviews(cpt, cpd_df, df)
                        if any(sentences.values()):
                            write_out_to_json(sentences, jsonfile)
                            if all(sentences.values()):
                                write_out_to_csv(sentences, csvfile)
                        else:
                            with open(exceptionfile, 'a+') as file:
                                file.write("general" + "=> empty sentences")
                    except Exception as msg:
                        print("write to ", exceptionfile)
                        with open(exceptionfile, 'a+') as file:
                            file.write("general" + "=> " + str(msg))
            else:
                print(exceptionfile, "  exits")
