import json 
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Process some files")
    parser.add_argument("--input", type=str, help='input filename to process')
    parser.add_argument("--output", type=str, help="output filename")

    parser.add_argument("--renovation_list", action="store_true", help="To clean the rennovation list from scrapper")
    parser.add_argument("--renovation_content", action="store_true", help="To clean the rennovation content from scrapper")


    return vars(parser.parse_args())



def extract_info_for_api_search(inputfile: str)-> None:
    """
    Extract info for api search from scrapped renovation hotel content.
    """

    with open(inputfile) as file:
        contents= json.load(file)
    
    info=[]
    for content in contents:
        # not dealing with zip because of uncertain format

        info.append({
            'id': content['id'],
            'name':content['hotelName'],
            'coordinates':"(" + content['latitude']+","+ content['longitude']+")" if content['latitude'] and content['longitude'] else " ",
            'phone':content['phone'] if content['phone'] else " ",
            'webpage':content['website'] if content['website'] else " ",
            'country': content['address'][-1] if content['address'] else " ",
            'street': content['address'][0] if content['address'] else " ",
        })

    return info



if __name__ == "__main__":
    args = parse_args()

    extract_info_for_api_search(args['input'])
