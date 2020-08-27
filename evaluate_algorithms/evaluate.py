import json
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import rootpath
import plac


def average_scores(eval_file):
    """
    Get overview of evaluations
    :param eval_file:
    :return: None
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


def average_score_per_category(eval_file):
    """
    Eval per category.
    :param eval_file: file to evaluate
    :return: None
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

    ### get dictionaries for each category
    eval = defaultdict(dict)

    for elem in data:
        category = elem['category']
        aerror = elem['annotation_error']
        hausdorrf = elem['hausdorff']
        precision = elem['precision']
        recall = elem['recall']
        f1 = elem['f1_score']
        randindex = elem['rand_index']
        gold_num = elem['gold_cpts_num']
        predicted_num = elem['predicted_cpts_num']

        if category not in eval:
            eval[category]= defaultdict(list)
            eval[category]['annotation error'].append(aerror)
            eval[category]['hausdorff'].append(hausdorrf)
            eval[category]['precision'].append(precision)
            eval[category]['recall'].append(recall)
            eval[category]['f1 score'].append(f1)
            eval[category]['rand index'].append(randindex)
            eval[category]['gold num'].append(gold_num)
            eval[category]['predicted num'].append(predicted_num)
            eval[category]['num'].append(1)
        else:
            eval[category]['annotation error'].append(aerror)
            eval[category]['hausdorff'].append(hausdorrf)
            eval[category]['precision'].append(precision)
            eval[category]['recall'].append(recall)
            eval[category]['f1 score'].append(f1)
            eval[category]['rand index'].append(randindex)
            eval[category]['gold num'].append(gold_num)
            eval[category]['predicted num'].append(predicted_num)
            eval[category]['num'].append(1)

    eval_avg = defaultdict(dict)
    for category, eval_metric in eval.items():
        eval_avg[category]={
            metric: round(np.mean(y),2) for metric,y in eval_metric.items() if metric !='num'
        }
        eval_avg[category]['num'] = sum(eval[category]['num'])

    print(eval_avg)

    eval_avg['file_num']= file_num

    method = algorithm + '_' + penalty

    return {method: eval_avg}

@plac.annotations(
    percategory = ("if the evaluation is per category",
                   "flag", "cat", bool)
)
def main(percategory=False):
    root_dir = rootpath.detect()
    data_dir = os.path.join(root_dir, 'evaluate_algorithms', 'data')
    eval_dict = {}

    eval_dict_per_category=defaultdict(dict)

    for filepath in Path(data_dir).rglob('*.json'):
        print(filepath)
        if percategory:
            eval = average_score_per_category(filepath)
            for method, category_eval in eval.items():
                for category, metric_dict in category_eval.items():
                    eval_dict_per_category[category][method] = metric_dict

        else:
            eval = average_scores(filepath)
            eval_dict.update(eval)
    if percategory:
        for category, eval_ in eval_dict_per_category.items():
            filename = category+'_eval.csv'
            df = pd.DataFrame.from_dict(eval_, orient='index')
            df.to_csv(os.path.join('eval_category', filename))
    else:
        df = pd.DataFrame.from_dict(eval_dict, orient='index')
        df.to_csv('eval.csv')


if __name__ == '__main__':
    plac.call(main)
