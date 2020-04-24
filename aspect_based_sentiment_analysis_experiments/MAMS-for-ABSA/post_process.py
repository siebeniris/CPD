import os
import pickle
import itertools
from xml.etree.ElementTree import parse
from collections import defaultdict
mode = 'term'

if mode =='category':

    with open('results','rb') as file:
        results = pickle.load(file)

    filepath='data/MAMS-ATSA/processed/test_ty_text.csv'

    preds = list(itertools.chain.from_iterable(results))
    print(len(preds))

    print(preds)

    import pandas as pd
    df = pd.read_csv(filepath)
    df['sentiment'] = preds

    df.to_csv('test_ty_sentiment.csv', index=False)

else:
    filename = '0a5c0a4c-36f7-46c4-9f13-91f52ba45ea5'
    eval_dir = '/home/yiyi/Documents/masterthesis/CPD/data/ABSA/eval'
    result_dir = '/home/yiyi/Documents/masterthesis/CPD/data/ABSA/results'
    aspect_extraction_dir = '/home/yiyi/Documents/masterthesis/CPD/data/aspect_extraction/output'

    with open(os.path.join(eval_dir, filename), 'rb') as file:
        eval= pickle.load(file)

    tree = parse(os.path.join(aspect_extraction_dir, filename+'.xml'))
    root = tree.getroot()

    preds = list(itertools.chain.from_iterable(eval))

    d = {
        'positive': 0,
        'negative': 1,
        'neutral': 2,
        'conflict': 3
    }
    d_ = {x: y for y,x in d.items()}
    print(d_)

    count = 0
    stats = defaultdict(int)
    for aspect in root.findall('.//aspectTerm'):
        prediction = d_[preds[count]]
        aspect.set('sentiment', prediction)
        stats[prediction] += 1
        count += 1

    assert count == len(preds)
    print(stats)
    print('writing the results to {}'.format(os.path.join(result_dir, filename+'.xml')))
    tree.write(os.path.join(result_dir, filename+'.xml'))