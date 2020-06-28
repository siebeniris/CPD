import os

import rootpath
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt


def get_cpd_df(file, aspect):
    """
    Get the dataframe for change point detection algorithm.
    :param file: file from "cpd_aspects"
    :param aspect: aspect from topics.
    :return: dataframe.
    """
    df = pd.read_csv(file)
    df.drop_duplicates(['sentence'], keep='first', inplace=True)  # drop duplicates based on sentence
    df = df.sort_values('date')

    df_new = df[df[aspect] == True]
    # df_new = df_new[df_new["renovation"] == True]

    print('length of df :', len(df))
    print('length of df_new', len(df_new))
    cpd_df = df_new.groupby(['date', 'uid'])['sentiment'].mean().reset_index()
    cpd_df = cpd_df.groupby('date')['sentiment'].mean().reset_index()

    print(cpd_df.head(3))

    # print("lenght of cpd_df:", len(cpd_df))
    return cpd_df, df_new


def binary_segmentation(cpd_df):
    """

    :param cpd_df:
    :param png_filepath:
    :return:
    """
    try:
        plt.rcParams.update({'figure.max_open_warning': 0})

        print("preview: ", cpd_df.head(3))
        sentiments = np.array(cpd_df.sentiment.to_list())
        print(sentiments)
        model = "rbf"  # find out why other models not applicable.

        N = len(sentiments)
        sigma = 0.5  # noise standard deviation
        algo = rpt.Binseg(model=model).fit(sentiments)
        bkps = algo.predict(pen=np.log(N) * sigma ** 2)
        # show results
        print(bkps)
        rpt.show.display(sentiments, bkps, figsize=(20, 10))
        plt.show()
        plt.cla()
        plt.close()
    except Exception as msg:
        print(msg)


def pelt_exact_segmentation(cpd_df):
    try:
        plt.rcParams.update({'figure.max_open_warning': 0})

        print("preview: ", cpd_df.head(3))
        sentiments = np.array(cpd_df.sentiment.to_list())
        print(sentiments)
        model = "ar"
        algo = rpt.Pelt(model=model, min_size=10, jump=5).fit(sentiments)
        bkps = algo.predict(pen=3)
        # show results
        print(bkps)
        fig, (ax,) = rpt.display(sentiments, bkps, figsize=(20, 10))
        plt.show()
        plt.cla()
        plt.close()

    except Exception as msg:
        print(msg)


def bottomUp_binary_segmentation(cpd_df):
    try:
        plt.rcParams.update({'figure.max_open_warning': 0})

        print("preview: ", cpd_df.head(3))
        sentiments = np.array(cpd_df.sentiment.to_list())
        print(sentiments)
        model = "rbf"  # l2 and ar. seems to be the best
        N = len(cpd_df)
        sigma = 0.5
        algo = rpt.BottomUp(model=model).fit(sentiments)
        bkps = algo.predict(pen=np.log(N) * sigma ** 2)
        # show results
        print(bkps)
        rpt.show.display(sentiments, bkps, figsize=(20, 10))
        plt.show()
        plt.cla()
        plt.close()

    except Exception as msg:
        print(msg)


def window_slider(cpd_df):
    try:
        plt.rcParams.update({'figure.max_open_warning': 0})

        print("preview: ", cpd_df.head(3))
        sentiments = np.exp2(cpd_df.sentiment.to_list())
        print(sentiments)
        model = "rbf"
        N = len(cpd_df)
        sigma = 0.5
        algo = rpt.Window(width=100, model=model).fit(sentiments)
        print(np.log(N) * sigma ** 2)
        bkps = algo.predict(pen=np.log(N) * sigma ** 2)

        # show results
        print(bkps)
        rpt.show.display(sentiments, bkps, [0,588, 825, 960, 1173], figsize=(20, 10))
        plt.show()
        plt.cla()
        plt.close()

    except Exception as msg:
        print(msg)


if __name__ == '__main__':
    import plac

    root_dir = rootpath.detect()

    test_file = 'data/cpd_aspects/164#41af2a52-407d-4c39-863f-57c6b3791920'
    testpath = os.path.join(root_dir, test_file)
    print(testpath)

    cpd_df, _ = get_cpd_df(testpath, "room")
    # binary_segmentation(cpd_df)
    # pelt_exact_segmentation(cpd_df)
    # bottomUp_binary_segmentation(cpd_df)
    window_slider(cpd_df)