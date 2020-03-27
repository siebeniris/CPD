import scrapy
import json
from scrapper.util import extract_info_for_api_search


class APIRequestNames(scrapy.Spider):
    name = 'apirequesthotels'

    def start_requests(self):
        # load the renovation hotel content from 'data' repo.
        filename = getattr(self, 'filename', None)
        # [{name:,coordinates, phone, webpage, country, street}]
        info = extract_info_for_api_search(filename)

        domain_link = 'http://api.trustyou.com/data/hotel/v1/clusters/search?text='

        print('there are {} hotels'.format(len(info)))
        for content in info:
            print(content)

            searched_text= 'name:'+ content['name']+'&coordinates:'+content['coordinates']\
                            +'&phone:'+content['phone']+'&country:'+content['country']\
                            +'&street:'+content['street'] +'&webpage:'+content['webpage']

            link = domain_link + searched_text + '&offset=0&limit=10'
            yield scrapy.Request(link, self.parse, meta={'name':content['name'], 'idx':content['id']})

    def parse(self, response):
        hotel_name = response.meta['name']
        idx = response.meta['idx']
        results = json.loads(response.body)
        yield {
            'searched': hotel_name,
            'id':idx,
            'results': results["response"]["data"]
        }