from skmultiflow.drift_detection.hddm_a import HDDM_A


def hoeffdings(cpd_df):
    """
    A drift detection method based on the Hoeffding's inequality.
    HDDM_A uses the average as estimator.

    :param cpd_df:
    :return:
    """
    try:
        data = cpd_df.sentiment
        hddm_a = HDDM_A(drift_confidence=0.4, two_side_option=True)
        LEN = len(data)

        detections = []

        for i, score in enumerate(data):
            hddm_a.add_element(score)
            if hddm_a.detected_change():
                detections.append(i)

        if len(detections) > 1:
            return detections + [LEN]
        else:
            return None
    except Exception:
        return None

