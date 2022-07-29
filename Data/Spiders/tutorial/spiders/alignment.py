import scrapy
print("strange")

with open(r'C:\Users\joshu\GitHub\safety-paper-classifier\Data\Links\alignment.txt') as f:
    alignmentPapers = f.readlines()
with open(r'C:\Users\joshu\GitHub\safety-paper-classifier\Data\Links\meta.txt') as f:
    metaPapers = f.readlines()

urls = alignmentPapers + metaPapers

class ArxivSpider(scrapy.Spider):
    name = 'alignmentspider'
    start_urls = urls
    def clean(self, text):
        text.replace('\n', ' ')
        text = text.strip()
        return text
    def parse(self, response):
        title = response.xpath('//*[@id="abs"]/h1/text()').get()
        abstract = self.clean(response.xpath('//blockquote[@class = "abstract mathjax"]/text()').getall()[1])
        yield {'title': title, 'abstract': abstract}


