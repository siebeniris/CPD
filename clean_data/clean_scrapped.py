import argparse
import ast
import json

import regex
from jellyfish import jaro_winkler


def parse_args():
    parser = argparse.ArgumentParser(description="Process some files")
    parser.add_argument("--input", type=str, help='input filename to process')
    parser.add_argument('--input1', type=str, help='input 1 filename to process')
    parser.add_argument("--output", type=str, help="output filename")

    parser.add_argument("--renovation_list", action="store_true", help="To clean the rennovation list from scrapper")
    parser.add_argument("--renovation_content", action="store_true", help="To clean the rennovation content from scrapper")
  
    parser.add_argument("--get_cluster_ids", action="store_true", help="To clean the rennovation content from scrapper")
    parser.add_argument("--renovation_file", type=str, help="input filename")
    parser.add_argument("--api_result", type=str, help="iutput filename")
    
    return vars(parser.parse_args())


def clean_scrapped_renovation_list(input, output):
    """
    clean scrapped rennovation list, to output to a jsonfile.
    """
    with open(input) as file:
        links = json.load(file)
    links_list = []
    for d in links:
        for k in d:
            link_list = d[k]
            links_list.extend(link_list)

    print("length:", len(links_list))
    with open(output, 'w')as file:
        json.dump(links_list, file)


def clean_scrapped_renovation_content(input, output):
    renovation_list = []
    with open(input) as file:
        idx = 0
        for line in file:
            content = json.loads(line)
            meta_ = content['meta'].strip()
            meta = ast.literal_eval(meta_)

            renovation_list.append({
                'id':idx,
                'title': content['title'],
                'meta': meta,
                'latitude': content['latitude'],
                'longitude': content['longitude'],
                'hotelName': content['hotelName'],
                'address': content['address'],
                'phone': content['phone'],
                'nrRooms': content['nrRooms'],
                'website': content['website'],
                'content': content["content"]
            })
            idx+=1

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


def gather_id_clusters(renovation_file, api_file, output):
    matches = []
    with open(renovation_file) as file:
        search = json.load(file)

    with open(api_file) as file:
        for line in file:
            result= json.loads(line)
            idx = result['id']
            match =[d for d in result['results'] if d['relevance']==1.0][0]
            searched_info = [d for d in search if d['id']==idx][0]
            madress = match['address']

            matches.append({
                'id': idx,
                'searched':{
                    'name': searched_info['hotelName'],
                    'address': searched_info['address'],
                    'coordinates':"(" + searched_info['longitude']+","+ searched_info['latitude']+")" if searched_info['latitude'] and searched_info['longitude'] else " ",
                    'phone': searched_info['phone'],
                    'webpage': searched_info['website']
                },
                'match': {
                    'cluster_id':match['cluster_id'],
                    'name':match['name'],
                    'address':[madress['street'], madress['city'],madress['state'], madress['zip'], madress['country']] ,
                    'coordinates':madress['coordinates'],
                    'phone':madress['phone'],
                    'webpage':madress['webpage'],
                    'deleted_on':match['deleted_on']
                }
            })

    with open(output, 'w') as file:
        json.dump(matches, file)
        

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
    if args["renovation_list"]:
        clean_scrapped_renovation_list(args["input"], args["output"])
    
    if args['renovation_content']:
        clean_scrapped_renovation_content(args['input'], args['output'])

    if args['get_cluster_ids']:
        gather_id_clusters(args['renovation_file'], args['api_result'], args['output'])

    # get_cluster_ids(args["input"], args["output"])
    #
    # extract_info_from_cleaned_list(args['input'], args['output'])
    # api_request_results(args['input'], args['output'])
    # get_cluster_ids(args['input'], args['input1'], args['output'])
    else:
        load_cluster_ids(args['input'], args['output'])