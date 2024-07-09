import requests
from bs4 import BeautifulSoup

r = requests.get('https://www.yelp.com/biz/ihop-daly-city')
soup = BeautifulSoup(r.text, 'html.parser')

review_spans = soup.select('p.comment__09f24__D0cxf span.raw__09f24__T4Ezm')

reviews = []
for span in review_spans:
    reviews.append(span.get_text())

for review in reviews:
    print(review)
