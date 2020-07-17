import pandas as pd
import numpy as np


def nonzero_len(l):
    ar = np.array(l)
    ar_ = ar[np.nonzero(ar)]
    return len(ar_)


def get_overview_stats(df):
    ys_2 = df[df['ys-na'] == 2]  # 1
    ya_2 = df[df['ya-ns'] == 2]  # 2
    ty_2 = df[df['two-yes'] == 2]  # 3
    tn_2 = df[df['two-no'] == 2]  # 4

    ys_1 = df[df['ys-na'] == 1]  # 1
    ya_1 = df[df['ya-ns'] == 1]  # 2
    ty_1 = df[df['two-yes'] == 1]  # 3
    tn_1 = df[df['two-no'] == 1]  # 4

    ys_len = nonzero_len(df['ys-na'])
    ya_len = nonzero_len(df['ya-ns'])
    ty_len = nonzero_len(df['two-yes'])
    tn_len = nonzero_len(df['two-no'])

    print("=============================")
    print("in total : ", len(df))
    print("yes sentiment, no aspect:", ys_len)
    print("yes aspect, no sentiment: ", ya_len)
    print("two yes:", ty_len)
    print("two no :", tn_len)

    print("=============================")
    print("agreed:")
    print("yes sentiment, no aspect:", len(ys_2))
    print("yes aspect, no sentiment: ",len(ya_2))
    print("two yes:", len(ty_2))
    print("two no :", len(tn_2))

    print("=============================")
    print("Disagreed: ")
    print("yes sentiment, no aspect:", len(ys_1))
    print("yes aspect, no sentiment: ", len(ya_1))
    print("two yes:", len(ty_1))
    print("two no :", len(tn_1))

    data_dict = df.to_dict('index')
    print(data_dict)

if __name__ == '__main__':
    annotated_sents_path = 'annotated_sentences_df.csv'
    annotated_sents_recat_path = 'annotated_sentences_df_recat.csv'

    df = pd.read_csv(annotated_sents_recat_path, index_col=0)
    get_overview_stats(df)
