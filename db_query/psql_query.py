'''
psql -h fdb.trustyou.com -U dev-ro -d ty_analytic -c
"copy (select uid, replace(array_to_string(token_array, ''), E'\n',' ') from hotel4x.review where cluster_id = '13e199d3-e843-4c87-894f-19da9caa5d2a') to stdout"
 > hotel-xyz.csv
'''
import argparse
import psycopg2
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Query reviews using cluster ids.")
    parser.add_argument("--input", type=str, help='input cluster ids')
    parser.add_argument("--output", type=str, help="output filename")
    return vars(parser.parse_args())


args = parse_args()
with open(args["input"]) as file:
    cluster_ids = json.load(file)


connection = psycopg2.connect(dbname="ty_analytic", user="dev-ro", host="fdb.trustyou.com")

query = """
    select uid, date_created, score, recommendation_score, title, lang, url, text from hotel4x.review where cluster_id=%(str)s
    ESCAPE ''
    """
cur = connection.cursor()

for idx in cluster_ids:
    print(idx)
    cur.execute(query, {'str': idx})
    cur.fetchall()
    filename = os.path.join(args['output'], idx)
    with open(filename, 'w') as file:
        json.dump(cur.fetchall(), file)


cur.close()
connection.close()
