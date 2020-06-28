import pandas as pd
import numpy as np


def get_cpd_df(file, aspect, renovation=False):
    """
    Get the dataframe for change point detection algorithm.
    1. drop the duplicates by "sentence", keep the first occurrence.
    2. sort the datafrmae by
    2. filter the dataframe with aspect /& renovation
    3.
    :param file: file from "cpd_aspects"
    :param aspect: aspect from topics.
    :param renovation: if the aspect of renovation should be applied.
    :return: change point detection dataframe, aspect/ & renovation category filtered dataframe.
    """
    print('loading the file :', file)
    df = pd.read_csv(file)
    df.drop_duplicates(['sentence'], keep='first', inplace=True)
    # drop duplicates based on sentence
    df = df.sort_values('date')

    df_new = df[df[aspect] == True]
    if renovation:
        df_new = df_new[df_new["renovation"] == True]

    print('length of df :', len(df))
    print('length of df_new', len(df_new))
    # aggregate by uid. score per review.
    cpd_df = df_new.groupby(['date', 'uid'])['sentiment'].mean().reset_index()
    # aggregate by date.
    cpd_df = cpd_df.groupby('date')['sentiment'].mean().reset_index()

    print('preview of change point detection dataframe: ')
    print(cpd_df.head(3))

    return cpd_df, df_new


def get_info_list(df):
    """
    Get the info from dataframe into List of (dates, uids, lemmas, sentences, sentiments).
    :param df: from sampled dataframe.
    :return:
    """
    df = df.sort_values('date')
    dates = df.date.to_list()
    uids = df.uid.to_list()
    lemmas = df.lemma.to_list()
    sentences = df.sentence.to_list()
    sentiments = df.sentiment.to_list()
    return list(zip(dates, uids, lemmas, sentences, sentiments))


def select_reviews(cpt, cpd_df, df, wbs=False):
    """
    Select reviews using cpt, cpd_df, new_df.
    :param cpt: a list of change points , e.g. [173, 225]
    :param cpd_df: dataframe for cpd (date, sentiment)
    :param df: new_df filtered by categories.
    :return: sentences, cpt, dates_periods, sentiment_mean
    """

    # total length of dates
    LEN = len(cpd_df)
    # only consider those sentences whose lemmas longer than 2.
    df = df[df['LEN'] > 2]
    print('df len > 2', len(df))
    # [0, 173, 225, 444]
    if wbs:
        cpt = sorted([0] + [int(x) for x in cpt] + [LEN])
    else:
        cpt =  sorted([0] + [int(x) for x in cpt] )
    # print(cpt)
    #
    # print('previews select reviews:')
    # print(cpd_df.head(3))
    # get the date periods.
    cpt_periods = list(zip(cpt[:-1], cpt[1:]))


    # [["2015-01-01", "2017-03-22"], ["2017-03-29", "2017-11-08"], ["2017-11-09", "2020-04-17"]]
    dates_periods = [(cpd_df.iloc[x].date, cpd_df.iloc[y - 1].date)
                     for x, y in cpt_periods]
    sentiment = cpd_df.sentiment.to_list()

    # [0.11313865595068318, -0.12056695491266556, 0.17144296948435583]
    sentiment_mean = [np.mean(sentiment[x:y]) for x, y in cpt_periods]
    print("dates periods: ")
    print(dates_periods)
    print("sentiment mean: ")
    print(sentiment_mean)

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
            if len(df_1) > 3:
                sentences[0] = get_info_list(df_1)

            df_2 = df_2[df_2['sentiment'] < sent2]
            if len(df_2) > 10:
                df_2 = df_2.sample(10)
            if len(df_2) > 3:
                sentences[1] = get_info_list(df_2)

        if trend == 1:
            df_1 = df_1[df_1['sentiment'] < sent1]
            if len(df_1) > 10:
                df_1 = df_1.sample(10)
            if len(df_1) > 3:
                sentences[0] = get_info_list(df_1)

            df_2 = df_2[df_2['sentiment'] > sent2]
            if len(df_2) > 10:
                df_2 = df_2.sample(10)
            if len(df_2) > 3:
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
            if len(df_first) > 3:
                sentences[0] = get_info_list(df_first)

        if first_trend == 1:
            df_first = df_first[df_first['sentiment'] < sent_first]
            if len(df_first) > 10:
                df_first = df_first.sample(10)
            if len(df_first) > 3:
                sentences[0] = get_info_list(df_first)

        for idx, sentiment_mean_ in enumerate(sentiment_mean_periods):
            min_, max_ = min(sentiment_mean_), max(sentiment_mean_)
            date_start, date_end = dates_periods[idx + 1]
            df_ = df[(df['date'] > date_start) & (df['date'] < date_end)]
            df_ = df_[(df_['sentiment'] > min_) & (df_['sentiment'] < max_)]

            if len(df_) > 10:
                df_ = df_.sample(10)
            if len(df_) > 3:
                sentences[idx + 1] = get_info_list(df_)

        # the last element.
        last = len(cpt_periods) - 1
        if last_trend == -1:
            df_last = df_last[df_last['sentiment'] > sent_last]
            if len(df_last) > 10:
                df_last = df_last.sample(10)
            if len(df_last) > 3:
                sentences[last] = get_info_list(df_last)

        if last_trend == 1:
            df_last = df_last[df_last['sentiment'] < sent_last]
            if len(df_last) > 10:
                df_last = df_last.sample(10)
            if len(df_last) > 3:
                sentences[last] = get_info_list(df_last)

    return sentences, cpt, dates_periods, sentiment_mean
