from skmultiflow.drift_detection import HDDM_W


def hoeffdings_wa(cpd_df):
    """
        A drift detection method based on the Hoeffding's inequality.
        HDDM_w uses the weighted average as estimator.

        :param cpd_df:
        :return:
        """
    try:

        data = cpd_df.sentiment
        hddm_w = HDDM_W(drift_confidence=0.4, lambda_option=0.03, two_side_option=True)
        LEN = len(data)

        detections = [0]

        for i, score in enumerate(data):
            hddm_w.add_element(score)
            if hddm_w.detected_change():
                detections.append(i)

        if len(detections) > 1:
            return detections + [LEN]
        else:
            return None

    except Exception:
        return None
