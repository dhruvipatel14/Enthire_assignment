import pandas as pd

pd.set_option('max_colwidth', 800)
import re
from nltk.stem.snowball import SnowballStemmer
import nltk
from string import punctuation
from nltk.corpus import stopwords

stopword = set(
    stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])


class Preprocess:

    def clean_text(self, text):
        #     text = re.sub(r'"',"",text)
        text = re.sub("@[\w]*", "", text)
        text = re.sub("https?://[A-Za-z0-9./]*", "", text)
        text = re.sub("\n", "", text)
        text = re.sub("\W+", " ", text)
        text = text.lower()
        return text

    def remove_stopwords(self, text):
        no_stopword_text = [w for w in text.split() if not w in stopword]
        return ' '.join(no_stopword_text)

    def stem_corpus(self, text):
        stemmer = SnowballStemmer("english")
        return stemmer.stem(text)

    def preprocessed_text(self, df):
        df['text'] = df['text'].apply(lambda x: self.clean_text(x))
        df['text'] = df['text'].apply(lambda x: self.remove_stopwords(x))
        df['text'] = df['text'].apply(lambda x: self.stem_corpus(x))

        return df
