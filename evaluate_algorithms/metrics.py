from ruptures.metrics import precision_recall, hausdorff, randindex


def precision_recall_f1_score(cpt1, cpt2, margin=20 ):
    """
    Caculate the precision, recall and f1 score.
    :param cpt1:
    :param cpt2:
    :param margin: allowed error , in sample numbers
    :return:
    """

    p, r = precision_recall(cpt1, cpt2, margin= margin)
    f1 = 2* p*r/(p+r)
    return p, r, f1

def hasudorff_distance(cpt1, cpt2):
    """
    Measures the worst prediction error.

    :param cpt1:
    :param cpt2:
    :return:
    """
    return hausdorff(cpt1, cpt2)


def rand_index(cpt1, cpt2):
    """
    Measure the similarity between two segmentations.
    :param cpt1:
    :param cpt2:
    :return:
    """
    return randindex(cpt1, cpt2)




