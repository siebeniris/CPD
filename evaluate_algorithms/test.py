import json
from collections import defaultdict

file = 'data/pelt_bic_20-07-23_eval.json'

with open(file) as reader:
    data = json.load(reader)

filenum = data['file_num']
evaluations = data['evaluations']

num_cat = defaultdict(int)

for eval in evaluations:
    category = eval['category']
    num_cat[category] += 1


with open('cat_dict.json', 'w') as writer:
    json.dump(num_cat, writer)