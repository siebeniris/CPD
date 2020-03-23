import json
import argparse
import os
import csv

from dateutil.parser import parse


def parse_args():
    """
    Parse Arguments
    :return: dict
    """
    parser = argparse.ArgumentParser(description="Process some files")
    parser.add_argument("--input", type=str, help='input filename/dir to process')
    parser.add_argument("--output", type=str, help="output filename/dir to write")
    parser.add_argument("--transform", action="store_true", help="if to transform")
    parser.add_argument("--inspect", action="store_true", help="if to inspect a file")
    parser.add_argument("--order", action="store_true", help="if to order dataframe")

    return vars(parser.parse_args())


def inspect_file(input):
    with open(input) as file:
        data = json.load(file)

    for item in data:
        print(item.keys())
        values = list(item.values())[0]
        type_= values[2]
        score = values[3]
        dateString = values[0]
        recommendation_rate = values[4]
        if recommendation_rate:
            # date = parse(dateString)
            print(recommendation_rate, type_, score)


def transform_json_to_csv(filepath, outputdir):
    """

    :param filename: the input filename
    :param outputdir: the directory of output
    :return: None
    """
    with open(filepath) as file:
        data = json.load(file)

    filename = os.path.basename(filepath)
    output_file = os.path.join(outputdir, filename)

    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",",
                                quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["uid", "date", "score", "recommendation_rate", "lang", "title", "text"])
        for item in data:
            uid = item[0]
            dateString = item[3]
            score = item[14]
            recommendation_rate = item[16]
            lang = item[11]
            title = item[6]
            text = ''.join(item[10]).replace("\n", "")

            if dateString:
                date = parse(dateString)
                csvwriter.writerow([uid, date, score, recommendation_rate, lang, title, text])


if __name__ == '__main__':
    args = parse_args()

    if args['inspect']:

        inspect_file(args['input'])

    if args['transform']:
        for file in os.listdir(args['input']):
            filepath = os.path.join(args['input'], file)
            transform_json_to_csv(filepath, args['output'])

