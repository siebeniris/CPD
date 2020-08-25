from skmultiflow.drift_detection import PageHinkley


def page_hinkley(cpd_df):
    """
    Page-Hinkley method for concept drift detection.
    :param cpd_df:
    :return:
    """

    try:
        data = cpd_df.sentiment
        LEN = len(data)
        DIST = int(LEN / 20)

        ph = PageHinkley(min_instances=DIST, delta=0.23, threshold=0.5)

        detections = [0]
        for i, score in enumerate(data):
            ph.add_element(score)
            if ph.detected_change():
                detections.append(i)

        if len(detections) > 1:
            return detections + [LEN]
        else:
            return None
    except Exception:
        return None

