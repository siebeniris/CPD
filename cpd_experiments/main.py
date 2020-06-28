import os
import json
import csv
import rootpath
from datetime import datetime

import plac

from prepare_cpd import get_cpd_df, get_info_list, select_reviews
from topics import topics as topic_dict
from write_out import write_out_to_json
# algorithms
from wbs import wild_binary_segmentation
from sbs import binary_segmentation


# (help, kind, abbrev, type, choices, metavar)
@plac.annotations(
    algorithm=("The name of the chosen algorithm", "positional", None, str,
               ["wbs", "sbs", "bbs", "pelt"]),
    renovation=("If the renovation category applies.", "flag", "reno", bool),
    penalty=("The name of the penalty function")
)
def main(
        algorithm,
        renovation=False,
        penalty="bic"
):
    root_dir = rootpath.detect()
    # sentiment analyzed, lemmas spell corrected, with aspects.
    cpd_aspects = os.path.join(root_dir, 'data', 'cpd_aspects')
    # define outputdir name as combined from algorithm and date.
    today = datetime.today().strftime('%y-%m-%d')
    outdir_name = algorithm +'_'+ penalty+ '_' + today
    print('algorithm', algorithm == "wbs")

    print(outdir_name)
    output_dir = os.path.join(root_dir, 'data', outdir_name)
    print('output_dir', output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

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
                    aspect_dir = os.path.join(output_dir, aspect)

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
                            if algorithm == "wbs":
                                cpt = wild_binary_segmentation(cpd_df, pngfile, wbs= True)
                                print("cpt: ", cpt)
                            if algorithm == "sbs":
                                cpt = binary_segmentation(cpd_df, pngfile, penalty=penalty)

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
