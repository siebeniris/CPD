import os
import json
import rootpath
from datetime import datetime

import numpy as np
import plac
from wcmatch import wcmatch

from prepare_cpd import get_cpd_df, get_info_list, select_reviews
from topics import topics as topic_dict
from write_out import write_out_to_json

# algorithms
from wbs import wild_binary_segmentation
from sbs import binary_segmentation
from pelt import pelt_exact_segmentation
from bbs import bottomUp_binary_segmentation
from window_based import window_slider
from rulsif import rulsifitting
from vonMisesFisher import von_mises_fisher


def files_to_evaluate(inputfile):
    """
    Load the data
    :param inputfile:
    :return:
    """
    with open(inputfile) as reader:
        cpts_gold = json.load(reader)
    return cpts_gold


# (help, kind, abbrev, type, choices, metavar)
@plac.annotations(
    algorithm=("The name of the chosen algorithm", "positional", None, str,
               ["wbs", "sbs", "bbs", "pelt", "window", "rulsif", "fisher"]),
    renovation=("If the renovation category applies.", "flag", "reno", bool),
    preeval=("If to generate stats for evaluation...", "flag", "preeval", bool),
    mean_feature=("If using von mises fisher using mean or sgd...", "flag", "mean_feature", bool),
    penalty=("The name of the penalty function", "option", "pen", str,
             ["bicl2", "aicl2", "bic"])

)
def main(
        algorithm,
        renovation=False,
        preeval=False,
        mean_feature=False,
        penalty="bic"
):
    root_dir = rootpath.detect()

    # sentiment analyzed, lemmas spell corrected, with aspects.
    cpd_aspects = os.path.join(root_dir, 'data', 'cpd_aspects')

    # define outputdir name as combined from algorithm and date.
    today = datetime.today().strftime('%y-%m-%d')
    outdir_name = algorithm + '_' + penalty + '_' + today
    print('algorithm', algorithm)

    print(outdir_name)
    output_dir = os.path.join(root_dir, 'data', outdir_name)
    print('output_dir', output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    if preeval:
        cpts_gold = files_to_evaluate(os.path.join(root_dir, 'data', 'gold_cpt_dict.json'))
        for fileId, d in cpts_gold.items():
            category = d['category']
            hotel_id = d['hotel_id']

            filepath = wcmatch.WcMatch(cpd_aspects, str(hotel_id)+'#*').match()[0]
            filename = os.path.basename(filepath)

            if os.path.isfile(filepath) and os.path.exists(filepath):
                # output
                aspect_dir = os.path.join(output_dir, category)
                if not os.path.exists(aspect_dir):
                    os.mkdir(aspect_dir)

                json_dir = os.path.join(aspect_dir, "json_file")
                # dir for storing cpt pictures.
                png_dir = os.path.join(aspect_dir, "pngs")
                if not os.path.exists(json_dir): os.mkdir(json_dir)
                if not os.path.exists(png_dir): os.mkdir(png_dir)

                jsonfile = os.path.join(json_dir, filename + '.json')
                pngfile = os.path.join(png_dir, filename + '.png')

                if category == 'renovation_room':
                    renovation = True
                    aspect = 'room'
                else:
                    renovation = False
                    aspect = category
                print(aspect, 'renovation: ', renovation)

                cpd_df, df = get_cpd_df(filepath, aspect, renovation=renovation)

                if len(cpd_df) > 0:
                    if algorithm == "wbs":
                        cpt = wild_binary_segmentation(cpd_df, pngfile)
                    if algorithm == "sbs":
                        cpt = binary_segmentation(cpd_df, pngfile, penalty=penalty)
                    if algorithm == "pelt":
                        cpt = pelt_exact_segmentation(cpd_df, pngfile, penalty=penalty)
                    if algorithm == 'bbs':
                        cpt = bottomUp_binary_segmentation(cpd_df, pngfile, penalty=penalty)
                    if algorithm == 'window':
                        cpt = window_slider(cpd_df, pngfile, penalty=penalty)

                    if algorithm =="rulsif":
                        cpt = rulsifitting(cpd_df, pngfile)

                    if algorithm == "fisher":
                        if mean_feature:
                            cpt = von_mises_fisher(cpd_df,pngfile, True)
                        else:
                            cpt = von_mises_fisher(cpd_df, pngfile, False)

                    print("cpt: ", cpt)
                    LEN = len(cpd_df)
                    try:
                        if cpt is not None:
                            if algorithm=='wbs':
                                cpt = sorted([0] + [int(x) for x in cpt] + [LEN])
                            else:
                                cpt = sorted([0] + [int(x) for x in cpt])

                            cpt_periods = list(zip(cpt[:-1], cpt[1:]))
                            dates_periods = [(cpd_df.iloc[x].date, cpd_df.iloc[y - 1].date)
                                             for x, y in cpt_periods]
                            sentiment = cpd_df.sentiment.to_list()
                            sentiment_mean = [np.mean(sentiment[x:y]) for x, y in cpt_periods]

                        if cpt:
                            d[algorithm] = {
                                'cpt': cpt,
                                'dates_periods': dates_periods,
                                'sentiment_mean': sentiment_mean
                            }

                            with open(jsonfile, 'w') as writer:
                                json.dump(d, writer)
                    except Exception:
                        print('does not work')
    else:
        aspects = list(topic_dict.keys())
        for aspect in aspects:
            # for aspect in ["room"]:
            if aspect != "renovation":
                # l = ['16#80710d34-34a5-4af8-a4bf-438ae18d3d06']
                for filename in os.listdir(cpd_aspects):
                    # for filename in l :
                    filepath = os.path.join(cpd_aspects, filename)
                    if os.path.isfile(filepath) and os.path.exists(filepath):
                        # print('Load file ', filepath)
                        # output aspect_dir

                        if renovation:
                            output_dir_ = os.path.join(output_dir, 'renovation-related')

                        else:
                            output_dir_ = output_dir

                        if not os.path.exists(output_dir_):
                            os.mkdir(output_dir_)

                        aspect_dir = os.path.join(output_dir_, aspect)

                        if not os.path.exists(aspect_dir):
                            os.mkdir(aspect_dir)

                        json_dir = os.path.join(aspect_dir, "json_file")
                        # dir for storing cpt pictures.
                        png_dir = os.path.join(aspect_dir, "pngs")
                        if not os.path.exists(json_dir): os.mkdir(json_dir)
                        if not os.path.exists(png_dir): os.mkdir(png_dir)

                        jsonfile = os.path.join(json_dir, filename + '.json')
                        pngfile = os.path.join(png_dir, filename + '.png')
                        exceptionfile = os.path.join(aspect_dir, filename)

                        if not os.path.exists(exceptionfile):
                            if not os.path.exists(jsonfile):
                                cpd_df, df = get_cpd_df(filepath, aspect, renovation=renovation)
                                if len(cpd_df) > 0:
                                    if algorithm == "wbs":
                                        cpt = wild_binary_segmentation(cpd_df, pngfile, wbs=True)
                                        print("cpt: ", cpt)
                                    if algorithm == "sbs":
                                        cpt = binary_segmentation(cpd_df, pngfile, penalty=penalty)
                                    if algorithm == "pelt":
                                        cpt = pelt_exact_segmentation(cpd_df, pngfile, penalty=penalty)
                                    # select the reviews:
                                    if cpt:
                                        try:
                                            print("change points detected :", cpt)
                                            sentences, cpt, dates_periods, sentiment_mean = select_reviews(cpt, cpd_df, df)

                                            if all(sentences.values()):
                                                print("len sentences: ", len(sentences))
                                                write_out_to_json(sentences, cpt, dates_periods, sentiment_mean, jsonfile)
                                            else:
                                                with open(exceptionfile, 'a+') as file:
                                                    file.write("general" + "=> empty sentences")
                                        except Exception as msg:
                                            print("write to ", exceptionfile)
                                            with open(exceptionfile, "a+") as file:
                                                file.write(aspect + '=>' + str(msg))

                        else:
                            print(exceptionfile, " exits")


if __name__ == '__main__':
    plac.call(main)
