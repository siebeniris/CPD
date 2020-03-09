"""
psql
==================================
uid                      0
type             	    1
source_uid              2
date                    3
date_created            4
author                  5
title        	        6  *
source_id                 7
url              	     8
mark_total               9
token_array          	  10  *
lang                	  11  *
cluster_id                 12
external_source_id         13
score                       14
token_array_ts_vector     15
recommendation_rate      16
original_recommendation_rate  17
respondable                 18
deleted_on                19
modified_on              20
duplicate_of              21
"""


import json
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Process some files")
    parser.add_argument("--input", type=str, help='input filename to process')
    parser.add_argument('--input1', type=str, help='input 1 filename to process')
    parser.add_argument("--output", type=str, help="output filename")

    return vars(parser.parse_args())

def clean_query_outputs(filename):
    """
    clean output for each file
    """
    with open(filename) as file:
        content_list = json.load(file)

    new_content_list =[]
    for content in content_list:
        uid = content[0]
        date = content[3]
        date_created = content[4]
        title = content[6]
        url = content[7]
        mark_total = content[9]

        text = ''.join(content[10])
        lang = content[11]
        score = content[14]
        recommendation_rate = content[16]

        new_content_list.append([uid, date, date_created, mark_total, score, recommendation_rate, lang, title, text])

    print(new_content_list)

if __name__ == '__main__':
    clean_query_outputs("query_outputs_all/77a35d4e-0d61-4390-ae86-5f8cf2bb9080")



