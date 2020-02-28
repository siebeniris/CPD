import argparse
import ast
import json
from jellyfish import jaro_winkler
import regex

def parse_args():
    parser = argparse.ArgumentParser(description="Process some files")
    parser.add_argument("--input", type=str, help='input filename to process')
    parser.add_argument('--input1', type=str, help='input 1 filename to process')
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
    renovation_list =[]
    with open(input) as file:
        for line in file:
            content = json.loads(line)
            # content = ast.parse(line.strip(), mode='eval')
            meta_ = content['meta'].strip()
            meta = ast.literal_eval(meta_)

            renovation_list.append({
                'title': content['title'],
                'meta': meta,
                'latitude': content['latitude'],
                'longitude': content['longitude'],
                'hotelName': content['hotelName'],
                'address': content['address'],
                'phone': content['phone'],
                'nrRooms': content['nrRooms'],
                'content': content["content"]
            })

    with open(output, 'w')as file:
        json.dump(renovation_list, file)


def api_request_results(input, output):
    apiresults={}
    with open(input) as file:
        for line in file:
            content= json.loads(line)
            apiresults[content['searched']] = content['results']
    with open(output, 'w') as file:
        json.dump(apiresults, file)

def extract_info_from_cleaned_list(input, output):
    with open(input) as file:
        ls = json.load(file)
    extracted_ls= []
    for item in ls:
        extracted_ls.append({
            'hotelName': item['hotelName'],
            'latitude': item['latitude'],
            'longitude': item['longitude'],
            'street': item['address'][0],
            'region': item['address'][1] if len(item['address']) >1 else None,
            'country': item['address'][-1],
            'phone': item['phone']

        })

    with open(output, 'w') as file:
        json.dump(extracted_ls, file)

def get_cluster_ids(input1, input2, output):
    searched_matched = list()
    with open(input1) as file:
        search = json.load(file)

    with open(input2) as file:
        results= json.load(file)

    for item in search:
        hotelname= item['hotelName']
        latitude = item['latitude']
        longitude = item['longitude']
        street = item['street'] if item['street'] else ''
        region = item['region'] if item['region'] else ''
        country = item['country'] if item['country'] else ''
        print(item['phone'])
        phone = ''.join(regex.findall(r'\d+',item['phone'])) if item['phone'] else ''

        scores =[]
        item_results = results.get(hotelname, [])
        if item_results:
            for result in item_results:
                found_name = result['name']
                found_street = result['address']['street'] if result['address']['street'] else ''
                found_region = result['address']['city']+', '+result['address']['zip'] if result['address']['city'] and result['address']['zip'] else ''
                found_country = result['address']['country'] if result['address']['country'] else ''
                found_phone = ''.join(regex.findall(r'\d+', result['address']['phone'])) if result['address']['phone'] else ''

                name_score = jaro_winkler(hotelname.lower(), found_name.lower())
                street_score = jaro_winkler(street.lower(), found_street.lower())
                region_score = jaro_winkler(region.lower(), found_region.lower())
                country_score = jaro_winkler(country.lower(), found_country.lower())
                phone_score = jaro_winkler(phone.strip(), found_phone.strip())

                if name_score > 0.95:
                    scores.append({
                        "cluster_id": result['cluster_id'],
                        "found_name":found_name,

                        "score":10
                    })
                else:
                    scores.append({
                        "cluster_id": result['cluster_id'],
                        "found_name": found_name,
                        "score": name_score + street_score + region_score + country_score + phone_score
                    })
        if scores:
            sorted_scores = sorted(scores, key=lambda k: k['score'], reverse=True)
            searched_matched.append({
                                'searched': hotelname,
                                'street': street,
                                'region': region,
                                'country': country,
                                'cluster_id':sorted_scores[0]['cluster_id'],
                                'found': sorted_scores[0]['found_name'],
                                'score':sorted_scores[0]['score']
                            })
    searched_matched_sorted = sorted(searched_matched, key=lambda k :k['score'])
    with open(output, 'w') as file:
        json.dump(searched_matched_sorted, file)

def load_cluster_ids(input, output):
    with open(input) as file:
        cluster_ids=json.load(file)

    cluster_id_list=[]
    for item in cluster_ids:
        if item['score']!=-2:
            cluster_id_list.append(
                item['cluster_id']
            )

    with open(output,'w') as file:
        json.dump(list(set(cluster_id_list)), file)


if __name__ == '__main__':
    args = parse_args()
    # clean_scrapped_renovation_list(args["input"], args["output"])
    # get_cluster_ids(args["input"], args["output"])
    # clean_scrapped_renovation_content(args['input'], args['output'])
    # extract_info_from_cleaned_list(args['input'], args['output'])
    # api_request_results(args['input'], args['output'])
    # get_cluster_ids(args['input'], args['input1'], args['output'])
    load_cluster_ids(args['input'], args['output'])