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
        """
        cleans the text with punctuation, @, and other special symbols
        :param text: single raw of text
        :return: cleaned text
        """
        #     text = re.sub(r'"',"",text)
        text = re.sub("@[\w]*", "", text)
        text = re.sub("https?://[A-Za-z0-9./]*", "", text)
        text = re.sub("\n", "", text)
        text = re.sub("\W+", " ", text)
        text = text.lower()
        return text

    def remove_stopwords(self, text):
        """
        Removes stop words from given text
        :param text: single raw of text
        :return: cleaned text
        """
        no_stopword_text = [w for w in text.split() if not w in stopword]
        return ' '.join(no_stopword_text)

    def stem_corpus(self, text):
        """
         Stemming of the given text
        :param text: single raw of text
        :return: stem text
        """
        stemmer = SnowballStemmer("english")
        return stemmer.stem(text)

    def preprocessed_text(self, df):
        """
        Pre process the given text data : removes stop word, clean text and
        stemming
        :param df: Dataframe
        :return: cleaned dataframe
        """
        df['text'] = df['text'].apply(lambda x: Preprocess.clean_text(self,x))
        df['text'] = df['text'].apply(lambda x: Preprocess.remove_stopwords(self,x))
        df['text'] = df['text'].apply(lambda x: Preprocess.stem_corpus(self,x))

        return df
