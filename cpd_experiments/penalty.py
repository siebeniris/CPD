import numpy as np

# for rupture library

def bic_l2_penalty(samples):
    """
    Get the bic l2 penalty from an array of samples
    :param samples:  an array of sentiment scores
    :return:
    """
    sigma = np.std(samples)
    print("sigma: ", sigma)
    N = len(samples)
    d = 1
    return sigma * sigma * np.log(N) * d

def aic_l2_penalty(samples):
    """

    :param samples:
    :return:
    """
    sigma = np.std(samples)
    d = 1
    return sigma*sigma*d


def bic_penalty(samples):
    """

    :param samples:
    :return:
    """
    sigma=np.std(samples)
    N = len(samples)
    d = 1
    return sigma* np.log(N)*d