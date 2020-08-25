import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import rootpath


def average_scores(eval_file):
    """
    Get overview of evaluations
    :param eval_file:
    :return:
    """
    filename = os.path.basename(eval_file)
    algorithm = filename.split('_')[0]
    if algorithm != 'wbs':
        penalty = filename.split('_')[1]
    else:
        penalty = ''

    with open(eval_file) as reader:
        eval_dict = json.load(reader)

    data = eval_dict['evaluations']
    file_num = eval_dict['file_num']


    aerror, haus, ps, rs, f1s, rands, gold_num, predicted_num = [], [], [], [], [], [],[],[]

    for ev in data:

        aerror.append(ev['annotation_error'])
        haus.append(ev['hausdorff'])
        ps.append(ev['precision'])
        rs.append(ev['recall'])
        f1s.append(ev['f1_score'])
        rands.append(ev['rand_index'])
        gold_num.append(ev['gold_cpts_num'])
        predicted_num.append(ev['predicted_cpts_num'])

    method = algorithm + '_' + penalty
    return {method:
        {
            "annotation error": round(np.mean(aerror),2),
            "precision": round(np.mean(ps), 2),
            "recall": round(np.mean(rs), 2),
            "f1 score": round(np.mean(f1s), 2),
            "hausdorff": round(np.mean(haus), 2),
            "rand index": round(np.mean(rands), 2),
            'avg gold num': round(np.mean(gold_num),1),
            'avg predicted num': round(np.mean(predicted_num),1),
            "detected %": round(file_num/356, 2)
        }}


if __name__ == '__main__':
    root_dir = rootpath.detect()
    data_dir = os.path.join(root_dir, 'evaluate_algorithms', 'data')
    eval_dict = {}
    for filepath in Path(data_dir).rglob('*.json'):
        print(filepath)
        eval = average_scores(filepath)
        eval_dict.update(eval)

    df = pd.DataFrame.from_dict(eval_dict, orient='index')
    df.to_csv('eval.csv')
