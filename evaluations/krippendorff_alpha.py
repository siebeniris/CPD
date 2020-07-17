'''
Python implementation of Krippendorff's alpha -- inter-rater reliability

(c)2011-17 Thomas Grill (http://grrrr.org)

Python version >= 2.4 required
'''
from itertools import chain

import numpy as np
import pandas as pd


def load_sents_units_dict(df):
    """
    Get the units dict from dataframe for sentences.
    :param df:
    :return:
    """
    ys = df['yes-sentiment']  # 1
    ya = df['yes-aspect']  # 2
    ns = df['no-sentiment']  # 3
    na = df['no-aspect']  # 4

    units = {}
    for id, x in enumerate(list(zip(ys, ya, ns, na))):
        ys_, ya_, ns_, na_ = x

        units[id] = ys_ * [1] + ya_ * [2] + ns_ * [3] + na_ * [4]
    return units


def load_sents_recat_units_dict(df):
    """
    Get the units dict from dataframe for sentences.
    :param df:
    :return:
    """
    ys = df['ys-na']  # 1
    ya = df['ya-ns']  # 2
    ty = df['two-yes']  # 3
    tn = df['two-no']  # 4

    units = {}
    for id, x in enumerate(list(zip(ys, ya, ty, tn))):
        ys_, ya_, ty_, tn_ = x
        l = [ys_, ya_, ty_, tn_]
        if ys_ == 1:
                l[0] += 1
        else:
                indx = np.random.randint(3)
                l[indx] += 1
        print(l)
        units[id] = list(chain.from_iterable([x*y for x,y in zip(l, [[1],[2],[3],[4]])]))
        # units[id] = ys_ * [1] + ya_ * [2] + ty_ * [3] + tn_ * [4]
    return units

def load_cpts_units_dict(df):
    """
    Get the units dict from dataframe.
    :param df:
    :return:
    """
    a_list = df['a']  # 1,
    b_list = df['b']  # 2
    c_list = df['c']  # 3
    total = df['Total']
    units = {}
    for id, x in enumerate(list(zip(a_list, b_list, c_list, total))):
        a, b, c, Total = x
        # if Total == 2:
        #     ind = [a, b, c].index(Total)
        #     elem = [1, 2, 3][ind]
        #     units[id] = [elem] * 3
        units[id] = a * [1] + b * [2] + c * [3]

    return units

def interval_metric(a, b):
    return (a-b)**2


def ratio_metric(a, b):
    return ((a-b)/(a+b))**2


def nominal_metric(a, b):
    return a != b


def krippendorff_alpha(units, metric=nominal_metric):
    """
    Calculate Krippendorff's alpha (inter-rater reliability)
    :param units:
    :param metric:
    :return:
    """
    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values

    print('units with pairable values')
    print(units)

    n = sum(len(pv) for pv in units.values())  # number of pairable values
    print(n)
    if n == 0:
        raise ValueError("No items to compare.")

    Do = 0.
    for grades in units.values():
        if metric:
            gr = np.asarray(grades)
            Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        else:
            Du = sum(metric(gi, gj) for gi in grades for gj in grades)
        Do += Du / float(len(grades) - 1)
    Do /= float(n)

    if Do == 0:
        return 1.

    De = 0.
    for g1 in units.values():
        if metric:
            d1 = np.asarray(g1)
            for g2 in units.values():
                De += sum(np.sum(metric(d1, gj)) for gj in g2)
        else:
            for g2 in units.values():
                De += sum(metric(gi, gj) for gi in g1 for gj in g2)
    De /= float(n * (n - 1))

    return 1. - Do / De if (Do and De) else 1.


if __name__ == '__main__':
    annotated_cpts_path = 'annotated_cpts_df.csv'
    df = pd.read_csv(annotated_cpts_path, index_col=0)
    units = load_cpts_units_dict(df)
    print(units)
    print("nominal metric: %.3f" % krippendorff_alpha(units))
    # 0.664, 1296 comparable pairs, 584 change points.

    # annotated_sents_path = 'annotated_sentences_df.csv'
    # df_sents = pd.read_csv(annotated_sents_path, index_col=0)
    # units_sents = load_sents_units_dict(df_sents)
    # print(units_sents)
    # 18610 comparable pairs, 9396 sentences.0,243 recat (two-yes, two-no, ys, ya)

    # normal comparable 37402, 9396 sentences, -0,117
    # print("nominal metric: %.3f" % krippendorff_alpha(units_sents)) #

