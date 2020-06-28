import rpy2.robjects as ro
from rpy2.robjects.packages import importr

wbs = importr("wbs")
grdevices = importr("grDevices")
rplot = ro.r('plot')


def wild_binary_segmentation(cpd_df, png_filepath):
    """
    Applying wild binary segmentation change-point-detection on sentiment scores.
    :param cpd_df: dataframe for change point detection algorithm
    :return: rupture change points.
    """
    try:
        print("preview:", cpd_df.head(2))
        sentiments = cpd_df.sentiment.to_list()
        sentiments_r = ro.FloatVector(sentiments)
        w = wbs.wbs(sentiments_r)
        w_cpt = wbs.changepoints(w)
        cpt = w_cpt.rx2("cpt.ic").rx2("ssic.penalty")
        cpt = list(cpt)
        print("cpt:", cpt)

        grdevices.png(file=png_filepath)
        rplot(w)
        grdevices.dev_off()
        return cpt
    except Exception as mg:
        print(mg)



