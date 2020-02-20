import argparse
import ast
import json
from jellyfish import jaro_winkler

def parse_args():
    parser = argparse.ArgumentParser(description="Process some files")
    parser.add_argument("--input", type=str, help='input filename to process')
    parser.add_argument("--output", type=str, help="output filename")

    return vars(parser.parse_args())

def clean_scrapped_renovation_list(input, output):
    links_list = []
    with open(input) as file:
        for line in file:
            if line.startswith("["):
                line = line.strip()
                link_list= ast.literal_eval(line)
                links_list.extend(link_list)
    print("length:", len(links_list))
    with open(output, 'w')as file:
        json.dump(links_list, file)


def clean_scrapped_renovation_content(input, output):
    renovation_name_list = []
    renovation_list =[]
    with open(input) as file:
        for line in file:
            content = ast.literal_eval(line)
            meta = ast.literal_eval(content["meta"])
            renovation_name_list.append(meta["author"]["name"])
            renovation_list.append({
                'title': content['title'],
                'meta': meta,
                'content': content["content"]
            })
    with open(output, 'w')as file:
        json.dump(renovation_list, file)


def get_cluster_ids(input, output):
    searched_matched = list()

    with open(input) as file:
        for line in file:
            scores = []
            # content = ast.literal_eval(line.strip())
            content = json.loads(line)
            searched_name = content["searched"]
            results = content["results"]
            # for result in results:
            #     found_name = result["name"]
            #     distance_score = jaro_winkler(searched_name.lower().strip(), found_name.lower().strip())
            #     scores.append({
            #         "cluster_id": result["cluster_id"],
            #         "distance": distance_score,
            #         "found_name": found_name
            #     })
            # sorted_scores = sorted(scores, key=lambda k: k['distance'], reverse=True)
            # print(sorted_scores[0], searched_name)
            selected = results[0]
            # if selected["distance"]> 0.85:
            score= jaro_winkler(searched_name.lower().strip(), selected["name"].lower().strip())
            print(score, searched_name, selected['name'])
            searched_matched.append({
                    'searched': searched_name,
                    'cluster_id': selected['cluster_id'],
                    'found': selected['name'],
                    'distance':score
                })

    searched_matched_sorted = sorted(searched_matched, key=lambda k :k['distance'])
    with open(output, 'w') as file:
        json.dump(searched_matched_sorted, file)




if __name__ == '__main__':
    args = parse_args()
    # clean_scrapped_renovation_list(args["input"], args["output"])
    get_cluster_ids(args["input"], args["output"])