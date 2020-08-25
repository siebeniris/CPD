# get all the emas png for the defined dataset for a overview.
import os
import rootpath
import json
from datetime import datetime

import matplotlib.pyplot as plt
from wcmatch import wcmatch

from prepare_cpd import get_cpd_df


def files_to_evaluate(inputfile):
    """
    Load the data_backup2
    :param inputfile:
    :return:
    """
    with open(inputfile) as reader:
        cpts_gold = json.load(reader)
    return cpts_gold


def emas(cpd_df, emas_png):
    """
    Get exponentially moving average photo from the dataframe.
    :param cpd_df: dataframe for change point detection.
    :param emas_png: output file to save the image.
    :return: None
    """
    # to validate the trend shown in wbs output
    # ema
    plt.rcParams.update({'figure.max_open_warning': 0})
    cpd_df.set_index('date', inplace=True)
    cpd_df['EMA_100'] = cpd_df['sentiment'].ewm(span=100, adjust=False).mean()
    cpd_df['EMA_50'] = cpd_df['sentiment'].ewm(span=50, adjust=False).mean()
    cpd_df['EMA_10'] = cpd_df['sentiment'].ewm(span=10, adjust=False).mean()
    plt.figure(figsize=[20, 10])
    plt.locator_params(axis='x', nbins=10)

    # plt.grid(True)
    plt.plot(cpd_df['sentiment'], label='score')
    plt.plot(cpd_df['EMA_10'], label='EMA-10')
    plt.plot(cpd_df['EMA_50'], label='EMA-50')
    plt.plot(cpd_df['EMA_100'], label='EMA-100')
    plt.legend(loc=2)

    plt.savefig(emas_png)
    plt.cla()
    plt.close()


if __name__ == '__main__':


    root_dir = rootpath.detect()

    cpd_aspects = os.path.join(root_dir, 'data_backup2', 'cpd_aspects')
    today = datetime.today().strftime('%y-%m-%d')
    outdir_name = 'emas_' + today
    output_dir = os.path.join(root_dir, 'data_backup2', outdir_name)
    print('output_dir', output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cpts_gold = files_to_evaluate(os.path.join(root_dir, 'data_backup2', 'gold_cpt_dict.json'))
    for fileId, d in cpts_gold.items():
        category = d['category']
        hotel_id = d['hotel_id']

        filepath = wcmatch.WcMatch(cpd_aspects, str(hotel_id) + '#*').match()[0]
        filename = os.path.basename(filepath)

        if os.path.isfile(filepath) and os.path.exists(filepath):
            # output
            aspect_dir = os.path.join(output_dir, category)
            if not os.path.exists(aspect_dir):
                os.mkdir(aspect_dir)

            emas_pngfile = os.path.join(aspect_dir, filename+'.png')

            if category=='renovation_room':
                renovation=True
                aspect='room'

            else:
                renovation=False
                aspect = category

            print(aspect,'renovation:', renovation)
            cpd_df, _ = get_cpd_df(filepath, aspect, renovation=renovation)
            emas(cpd_df, emas_pngfile)


