from skmultiflow.drift_detection import DDM


def drift_detection_method(cpd_df):
    """
    Page-Hinkley method for concept drift detection.
    :param cpd_df:
    :return:
    """
    try:
        data = cpd_df.sentiment
        LEN = len(data)
        DIST = int(LEN / 20)

        ddm = DDM(min_num_instances=DIST, out_control_level=2)

        detections = [0]
        for i, score in enumerate(data):
            ddm.add_element(score)
            if ddm.detected_change():
                detections.append(i)

        if len(detections) > 1:
            return detections + [LEN]
        else:
            return None

    except Exception:
        return None

