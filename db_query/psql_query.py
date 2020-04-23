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

not_matched = [2, 44, 49, 84, 138, 177, 181, 245, 274, 302, 324, 392, 403, 430]

connection = psycopg2.connect(dbname="ty_analytic", user="dev-ro", host="fdb.trustyou.com")

query = """
    select * from hotel4x.review where cluster_id=%(str)s
    """
cur = connection.cursor()

for idx, filename in cluster_ids.items():
    if int(idx) not in not_matched:
        print(filename)
        cur.execute(query, {'str': filename})
        filepath = os.path.join(args['output'], idx+"#"+filename)
        with open(filepath, 'w') as file:
            json.dump(cur.fetchall(), file, default=str)


cur.close()
connection.close()
