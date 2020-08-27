import os
import json
from pathlib import Path

import plac
import rootpath

from metrics import annotation_error, precision_recall_f1_score, hasudorff_distance, rand_index


def evaluate_algorithm(file_path, algorithm):
    with open(file_path) as file:
        data = json.load(file)

    gold_cpts = data['valid_cpts']
    category = data['category']
    predicted = data[algorithm]['cpt']
    hotel_id = data['hotel_id']


    predicted = sorted(list(set(predicted)))

    aerror = annotation_error(gold_cpts, predicted)
    hasudorff = hasudorff_distance(gold_cpts, predicted)
    p, r, f1 = precision_recall_f1_score(gold_cpts, predicted)
    randIndex = rand_index(gold_cpts, predicted)

    fileId = os.path.basename(file_path).split('#')[0]

    gold_cpts_num  = len(gold_cpts)-2
    predicted_num = len(predicted) -2

    return {
        'fileId': fileId,
        'hotel_id': hotel_id,
        'category': category,
        'gold_cpts': gold_cpts,
        algorithm + '_cpts': predicted,
        "annotation_error": aerror,
        'hausdorff': hasudorff,
        'precision': p,
        'recall': r,
        'f1_score': f1,
        'rand_index': randIndex,
        "gold_cpts_num": gold_cpts_num,
        "predicted_cpts_num": predicted_num
    }


# @plac.annotations(dirname=('dirname', 'option', "", str))
def main():
    root_dir = rootpath.detect()
    data_dir = os.path.join(root_dir, 'data', 'cpd_algorithms')

    cwd = os.getcwd()

    for dirname in os.listdir(data_dir):
        print(dirname)

        output_file = os.path.join(cwd, "data", dirname + '_eval.json')
        if not os.path.exists(output_file):
            try:

                result_dir = os.path.join(data_dir, dirname)
                algorithm = dirname.split('_')[0]
                evaluations = list()

                file_num, gold_cpts_num, predicted_cpts = 0, 0, 0

                for filepath in Path(result_dir).rglob('*.json'):
                    eval = evaluate_algorithm(filepath, algorithm)


                    evaluations.append(eval)
                    file_num += 1

                eval_dict = {
                    "file_num": file_num,
                    "evaluations": evaluations
                }
                output_file = os.path.join(cwd, "data", dirname + '_eval.json')
                with open(output_file, 'w') as writer:
                    json.dump(eval_dict, writer)
            except Exception:
                print('doesnt work ...')


if __name__ == '__main__':
    # dirn = 'bbs_bicl2_20-07-23'
    plac.call(main)
