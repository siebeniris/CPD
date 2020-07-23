import numpy as np



# https://towardsdatascience.com/inter-annotator-agreement-2f46c6d37bf3

'''
cohen's kappa is a statistic to measure the reliability between annotators
for qualitative(categorical) items.
it is a more robust measure than simple percent agreement calculations.
as k takes into account the possibility of the agreement occuring by chance.
it is a pairwise reliability measure between two annotators.
Cohen’s kappa statistic is the agreement between two raters where Po is the
 relative observed agreement among raters (identical to the accuracy),
 and Pe is the hypothetical probability of chance agreement.
 Below you will find the programmatic implementation of this evaluation metric.
'''

def cohen_kappa(ann1, ann2):
    """Computes Cohen kappa for pair-wise annotators.

    :param ann1: annotations provided by first annotator
    :type ann1: list
    :param ann2: annotations provided by second annotator
    :type ann2: list

    :rtype: float
    :return: Cohen kappa statistic
    """
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (Po)

    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count

    return round((A - E) / (1 - E), 4)



def fleiss_kappa(M):
    """Computes Fleiss' kappa for group of annotators.

    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix

    :rtype: float
    :return: Fleiss' kappa score
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators
    tot_annotations = N * n_annotators  # the total # of annotations
    category_sum = np.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    PbarE = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide by amount of items

    return round((Pbar - PbarE) / (1 - PbarE), 4)