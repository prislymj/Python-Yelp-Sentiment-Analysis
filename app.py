import requests
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import textblob
from textblob import Word
from textblob import TextBlob

nltk.download('stopwords')

r = requests.get('https://www.yelp.com/biz/ihop-daly-city')
soup = BeautifulSoup(r.text, 'html.parser')

review_spans = soup.select('p.comment__09f24__D0cxf span.raw__09f24__T4Ezm')
reviews = [span.get_text() for span in review_spans]
#analysing data
df = pd.DataFrame(reviews, columns=['review'])

df['word_count'] = df['review'].apply(lambda x: len(x.split()))
df['char_count'] = df['review'].apply(lambda x: len(x))
df['average_word_length'] = df['review'].apply(lambda x: np.mean([len(word) for word in x.split()]))

stop_words = set(stopwords.words('english'))
df['stopword_count'] = df['review'].apply(lambda x: len([word for word in x.split() if word.lower() in stop_words]))
df['stopword_rate'] = df['stopword_count'] / df['word_count']

#cleaning data
df['lowercase']=df['review'].apply(lambda x: " ".join(word.lower() for word in x.split()))
df['punctuation']=df['lowercase'].str.replace('[^\w\s]','')
df['stopwords']=df['punctuation'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))

pd.Series(" ".join(df['stopwords']).split()).value_counts()[:30]
other_stop_words = ['get','told','came','went','one','us','also','even','got','said','asked','back','order','ordered','went','like','would','could','eat','go','make','take','know','want','see','need','come','try','give','look']

df['cleanreview']=df['stopwords'] = df['stopwords'].apply(lambda x: " ".join(word for word in x.split() if word not in other_stop_words))
pd.Series(" ".join(df['cleanreview']).split()).value_counts()[:30]

#lemmatization
df['lemmatized']=df['cleanreview'].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))           

#sentiment analysis
df['polarity'] = df['lemmatized'].apply(lambda x: TextBlob(x).sentiment[0])
df['subjectivity'] = df['lemmatized'].apply(lambda x: TextBlob(x).sentiment[1])

df.drop(['lowercase', 'punctuation', 'stopwords', 'cleanreview'], axis=1, inplace=True)

# Classify reviews
def classify_polarity(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['polarity'].apply(classify_polarity)

print("\nSentiment Classification:")
print(df['sentiment'].value_counts())

print(df.head())