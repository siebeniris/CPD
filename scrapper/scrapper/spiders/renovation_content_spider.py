import scrapy
import json



class RenovationContentSpider(scrapy.Spider):
    name = "renovationContent"

    def start_requests(self):
        filename = getattr(self, 'filename', None)
        with open(filename) as file:
            urls = json.load(file)
        domain_link = 'https://www.hospitalitynet.org'
        for url in urls:
            link = domain_link+url
            yield scrapy.Request(link, self.parse)

    def parse(self, response):
        yield{
            'title': response.xpath('//title/text()').get(),
            'meta':response.xpath('//script[@type="application/ld+json"]/text()').get(),
            'latitude': response.xpath('//div[@id="map"]/@data-latitude').get(),
            'longitude':response.xpath('//div[@id="map"]/@data-longitude').get(),
            'hotelName': response.css('aside.bg-white header::text').get(),
            'address': response.css('aside.bg-white p::text').getall(),
            'phone': response.css('aside.bg-white span.phone::text').get(),
            'nrRooms': response.css('aside.bg-white span.bg-gray-100::text').get(),
            'website': response.css("a.block.text-white::attr(href)").get(),
            'content': response.css('div.content *::text').getall()
        }

