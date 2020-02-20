import scrapy
import json

class APIRequestNames(scrapy.Spider):
    name = 'apirequestnames'

    def start_requests(self):
        filename = getattr(self, 'filename', None)
        with open(filename) as file:
            names = json.load(file)
        domain_link = 'http://api.trustyou.com/data/hotel/v1/clusters/search?text='
        for name in names:
            link = domain_link + name + '&offset=0&limit=10'
            yield scrapy.Request(link, self.parse, meta={'hotel_name':name})

    def parse(self, response):
        hotel_name = response.meta['hotel_name']
        results = json.loads(response.body)
        yield {
            'searched': hotel_name,
            'results': results["response"]["data"]
        }