import matplotlib.pyplot as plt


def emas(cpd_df, emas_png):
    """
    Get exponentially moving average photo from the dataframe.
    :param cpd_df: dataframe for change point detection.
    :param emas_png: output file to save the image.
    :return: None
    """
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
    plt.plot(cpd_df['EMA_50'], label='EMA-50')
    plt.plot(cpd_df['EMA_100'], label='EMA-100')
    plt.legend(loc=2)
    plt.savefig(emas_png)
    plt.cla()
    plt.close()