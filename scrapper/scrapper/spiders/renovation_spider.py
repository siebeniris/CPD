import scrapy

class RenovationSpider(scrapy.Spider):
    name = 'renovation_list'

    def start_requests(self):
        urls = ['https://www.hospitalitynet.org/list/1-12/hotel-renovations.html']

        for url in urls:
            yield scrapy.Request(url= url, callback=self.parse)

    def parse(self, response):
        link = response.css('div.bg-white.mb-2 a.gaa::attr(href)').extract()
        print(link)
        yield link

        domain_link = 'https://www.hospitalitynet.org'
        next_page = response.css('a.next::attr(href)').get()
        print(next_page)
        if next_page:
            page = domain_link+next_page
            yield response.follow(page, self.parse)




