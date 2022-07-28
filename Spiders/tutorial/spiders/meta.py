with open(r'C:\Users\joshu\GitHub\safety-paper-classifier\Data\meta.txt') as f:
    metaPapers = f.readlines()

urls = metaPapers
import scrapy
class ArxivSpider(scrapy.Spider):
    name = 'metaspider'
    start_urls = urls
    def clean(self, text):
        text.replace('\n', ' ')
        text = text.strip()
        return text
    def parse(self, response):
        title = response.xpath('//*[@id="abs"]/h1/text()').get()
        abstract = self.clean(response.xpath('//blockquote[@class = "abstract mathjax"]/text()').getall()[1])
        yield {'title': title, 'abstract': abstract}
