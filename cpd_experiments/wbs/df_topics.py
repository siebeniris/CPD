import os
import swifter
import json
import csv
import rootpath
from tqdm import tqdm

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from topics import topics as topic_dict
from timer import Timer

def get_df_topic(filename):
    """
    Get the df with topics.
    :param file: filename from input dir.
    :return: None
    """
    root_dir = rootpath.detect()
    aspect_dir = os.path.join(root_dir, 'data_backup2', 'cpd_aspects')
    input_dir = os.path.join(root_dir, 'data_backup2', 'spellchecked')
    filepath = os.path.join(input_dir, filename)
    outputfile = os.path.join(aspect_dir, filename)

    if os.path.isfile(filepath) and not os.path.exists(outputfile):
        print("loading file ", filepath)
        timer = Timer()
        timer.start()
        try:
            df = pd.read_csv(filepath)
            df['date'] = df['date'].astype('datetime64')
            df['sentiment'] = df['sentiment'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
            df['lemma'] = [json.loads(x) for x in df['lemma']]
            df['LEN'] = [len(x) for x in df['lemma']]

            for keyword, values in topic_dict.items():
                df[keyword] = df.apply(lambda x: sum([y in x.lemma for y in values]) > 0,
                                               axis=1)
            df = df.sort_values('date')
            print('output file to ', outputfile)
            df.to_csv(outputfile)
        except Exception as msg:
            print(msg)

            print("Exception raised ", outputfile)
        timer.stop()

if __name__ == '__main__':
    root_dir = rootpath.detect()
    input_dir = os.path.join(root_dir, 'data_backup2', 'spellchecked')
    filenames = [filename for filename in os.listdir(input_dir)]
    print("filenames :" , filenames[:3])
    Parallel(n_jobs=3)(delayed(get_df_topic)(filename) for filename in filenames)

