import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt

from penalty import bic_l2_penalty, aic_l2_penalty, bic_penalty


# get a range of all the data_backup2.
# choose a smaller penalty value.
def window_slider(cpd_df, png_filepath, penalty):
    """
    Use window_based search method.
    :param cpd_df:
    :param png_filepath:
    :return:
    """
    try:
        plt.rcParams.update({'figure.max_open_warning': 0})

        print("preview: ", cpd_df.head(3))
        sentiments = np.array(cpd_df.sentiment.to_list())
        model = "rbf"  # find out why other models not applicable.

        if penalty == "bicl2":
            pen = bic_l2_penalty(sentiments)
            print(pen)
        if penalty == "aicl2":
            pen = aic_l2_penalty(sentiments)
            print(pen)
        if penalty == "bic":
            pen = bic_penalty(sentiments)
            print(pen)

        algo = rpt.Window(width=20, model=model).fit(sentiments)
        bkps = algo.predict(pen=pen)
        # show results
        print(bkps)
        rpt.show.display(sentiments, bkps, figsize=(20, 10))
        plt.savefig(png_filepath)
        plt.cla()
        plt.close()
        return bkps
    except Exception as msg:
        print(msg)
