#################################################################################################
## https://techrando.com/2019/08/14/a-brief-introduction-to-change-point-detection-using-python/#
#################################################################################################

import argparse
import os
import json

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt
import time

def parse_args():
    """
    Parse Arguments
    :return: dict
    """
    parser = argparse.ArgumentParser(description="Process some files")
    parser.add_argument("--input", type=str, help='input filename/dir to process')
    parser.add_argument("--output", type=str, help="output filename/dir to write")
    parser.add_argument("--pelt", action="store_true", help="pelt search method")
    parser.add_argument("--window", action="store_true", help="window search method")
    parser.add_argument("--binseg", action="store_true", help="binary segmentation search method")
    parser.add_argument("--dynp", action="store_true", help="dynamic programming search method")
    parser.add_argument("--batch", action="store_true", help="work in batch")

    return vars(parser.parse_args())


def pelt(scores, outputfile):
    # slow.
    model = "rbf"
    algo = rpt.Pelt(model=model).fit(scores)
    result = algo.predict(pen=10)
    print(result)
    rpt.display(scores, result, figsize=(10, 6))
    plt.title('Change Point Detection: Pelt Search Method')
    plt.savefig(outputfile)


def binseg(scores, outputfile, stats):
    # fast
    model = "l2"
    n_samples = scores.shape[0]
    algo = rpt.Binseg(model=model).fit(scores)
    try:
        my_bkps = algo.predict(n_bkps=5)
        with open(stats, 'w') as file:
            json.dump([my_bkps,n_samples ], file)
        # show results

        rpt.show.display(scores, my_bkps, figsize=(10, 6))
        plt.title('Change Point Detection: Binary Segmentation Search Method')
        plt.savefig(outputfile)
        plt.close()
    except Exception:
        print(Exception)


def window(scores, outputfile, stats):
    # http://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/detection/window.html
    # fast
    model = "rbf"
    n_samples = scores.shape[0]
    # sigma = 5
    algo = rpt.Window(width=50, model=model).fit(scores)
    # my_bkps = algo.predict(pen=np.log(n_samples)*sigma)
    try:
        my_bkps = algo.predict(n_bkps=5)
        with open(stats, 'w') as file:
            json.dump([my_bkps, n_samples], file)
        rpt.show.display(scores, my_bkps, figsize=(10, 6))
        plt.title('Change Point Detection: Window-Based Search Method')
        plt.savefig(outputfile)
        plt.close()
    except Exception:
        print(Exception)


def dynp(scores, outputfile):
    # slow
    model = "l1"
    algo = rpt.Dynp(model=model, min_size=3, jump=5).fit(scores)
    my_bkps = algo.predict(n_bkps=5)
    print(my_bkps)
    rpt.show.display(scores, my_bkps, figsize=(10, 6))
    plt.title('Change Point Detection: Dynamic Programming Search Method')
    plt.savefig(outputfile)


if __name__ == '__main__':
    args = parse_args()
    if args['batch']:
        start_time = time.time()

        for file in os.listdir(args['input']):

            filepath = os.path.join(args['input'], file)
            # read the file into dataframe
            df = pd.read_csv(filepath)

            # order the df by date
            df = df.sort_values('date')

            # scores
            scores = np.array(df['score'])

            #  rupture_result/djfkdjfkdjkf/, for each sample.
            output_dir = os.path.join(args['output'], file)



            if not os.path.exists(output_dir):
                os.mkdir(output_dir)


            # if args['pelt']:
            #     outputfile = os.path.join(output_dir, 'plt')
            #     pelt(scores, outputfile)

            # if args['dynp']:
            #     outputfile = os.path.join(output_dir, 'dynp')
            #     dynp(scores, outputfile)

            if args['binseg']:
                print('processing file:', file)
                outputfile = os.path.join(output_dir,'binseg')
                stats= os.path.join(output_dir,'binseg.json')
                binseg(scores, outputfile,stats)

            if args['window']:
                print('processing file:', file)
                outputfile = os.path.join(output_dir, 'window')
                stats= os.path.join(output_dir,'window.json')

                window(scores, outputfile, stats)
        end_time = time.time()
        print('Processing time:', end_time - start_time)

    else:
        df = pd.read_csv(args['input'])
        filename = os.path.basename(args['input'])
        # order the df by date
        df = df.sort_values('date')

        # scores
        scores = np.array(df['score'])

        output_dir = os.path.join(args['output'], filename)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # if args['pelt']:
        #     outputfile = os.path.join(output_dir, 'plt')
        #     pelt(scores, outputfile)

        if args['binseg']:
            outputfile = os.path.join(output_dir, 'binseg')
            stats = os.path.join(output_dir, 'binseg.json')
            binseg(scores, outputfile, stats)

        if args['window']:
                outputfile = os.path.join(output_dir, 'window')
                stats = os.path.join(output_dir, 'window.json')
                window(scores, outputfile, stats)

        # if args['dynp']:
            #     outputfile = os.path.join(output_dir, 'dynp')
            #     dynp(scores, outputfile)