import requests
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

nltk.download('stopwords')

r = requests.get('https://www.yelp.com/biz/ihop-daly-city')
soup = BeautifulSoup(r.text, 'html.parser')

review_spans = soup.select('p.comment__09f24__D0cxf span.raw__09f24__T4Ezm')
reviews = [span.get_text() for span in review_spans]

df = pd.DataFrame(reviews, columns=['review'])

df['word_count'] = df['review'].apply(lambda x: len(x.split()))
df['char_count'] = df['review'].apply(lambda x: len(x))
df['average_word_length'] = df['review'].apply(lambda x: np.mean([len(word) for word in x.split()]))

stop_words = set(stopwords.words('english'))
df['stopword_count'] = df['review'].apply(lambda x: len([word for word in x.split() if word.lower() in stop_words]))
df['stopword_rate'] = df['stopword_count'] / df['word_count']

print(df.head(5))
