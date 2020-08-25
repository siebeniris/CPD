from skmultiflow.drift_detection import KSWIN


def kolmogorov_smirnov_windowing(cpd_df):
    """
    KSWIN is a concept change detection method based on the
    Kolmogorov-Smirnov statistical test. KS-test is a statistical
    test with no assumption of underlying data_backup2 distribution.
    KSWIN maintains a sliding window of fixed size n (window size)
    The last r (Stat_size) sample of the window ...
    :param cpd_df: df for change point detection
    :return:
    """
    try:

        data = cpd_df.sentiment
        LEN = len(data)
        MIN_DIST = int(LEN / 20)
        WIN_SIZE = int(LEN / 10)

        kswin = KSWIN(alpha=0.005, window_size=WIN_SIZE, stat_size=MIN_DIST)
        detections = []

        for i, score in enumerate(data):
            kswin.add_element(score)
            if kswin.detected_change():
                detections.append(i)

        if len(detections) > 0:

            return detections
        else:
            return None

    except Exception:
        return None
