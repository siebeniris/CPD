import numpy as np
import matplotlib.pylab as plt
import peakutils

from density_ratio_estimation.change_point_detector.density_ratio_estimator import DRChangeRateEstimator


def von_mises_fisher(cpd_df, png_filepath, mean_feature=True):
    """
    Rulsifitting.
    :param cpd_df:
    :param png_filepath:
    :param feature: mean or svg
    :return:
    """
    plt.rcParams.update({'figure.max_open_warning': 0})
    data_x = cpd_df.sentiment
    LEN = len(data_x)

    MIN_DIST = int(LEN / 10)
    WIN_SIZE= int(LEN/20)

    detector = DRChangeRateEstimator(
        sliding_window=WIN_SIZE,
        pside_len=WIN_SIZE, # past side length
        cside_len=WIN_SIZE, # current side klength
        mergin=0,  # mergin between past side and current side
        trow_offset=0,
        tcol_offset=0
    )
    if mean_feature:
        detector.build(estimation_method="von_mises_fisher",
                       options=detector.MEAN_OPTION)
    else:
        detector.build(estimation_method="von_mises_fisher",
                       options=detector.SVD_OPTION)


    change_rates = detector.transform(data_x)
    change_rates = np.nan_to_num(change_rates)

    peak_indexes = peakutils.indexes(change_rates, thres=0.1, min_dist=MIN_DIST)
    print('peak indexes:', peak_indexes)
    if len(peak_indexes)>0:
        plt.plot(peak_indexes,
                 change_rates[peak_indexes],
                 "o", color="red")
        plt.plot(change_rates)
        plt.savefig(png_filepath)
        plt.cla()
        plt.close()

        cpts = sorted(list(peak_indexes) + [len(data_x)])
        return cpts
    else:
        return None