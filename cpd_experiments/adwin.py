from skmultiflow.drift_detection import ADWIN


def ad_win(cpd_df):
    """
    Page-Hinkley method for concept drift detection.
    :param cpd_df:
    :return:
    """
    try:
        data = cpd_df.sentiment

        adwin = ADWIN(delta=0.5)
        detections = []
        for i, score in enumerate(data):
            adwin.add_element(score)
            if adwin.detected_change():
                detections.append(i)


        if len(detections) > 0:
            return detections
        else:
            return None
    except Exception:
        return None