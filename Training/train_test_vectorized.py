from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import class_weight
import numpy as np

class Pretraining:
    def __init__(self,df):
        self.df =  df

    def dataset_split(self):
        x_train, x_test, y_train, y_test = train_test_split(self.df['text'],
                                                            self.df[
                                                                'airline_sentiment'],
                                                            test_size=0.2,
                                                            random_state=42)
        print('x_train,x_test', x_train.shape, x_test.shape)
        print('y_train,y_test', y_train.shape, y_test.shape)

        return x_train,x_test,y_train,y_test

    def tfidf(self):
        x_train, x_test, y_train, y_test = Pretraining.dataset_split(self)
        tf_idf_vectorizer = TfidfVectorizer(lowercase=True,stop_words=None,
                                            max_df = 0.75,max_features=1000,
                               ngram_range=(1,2))
        x_train_vectorized = tf_idf_vectorizer.fit_transform(x_train)
        x_test_vectorized = tf_idf_vectorizer.transform(x_test)

        print('----------------------class distribution of '
              'sentiments---------------------')
        print(y_train.value_counts())

        print('--------------------------------------class '
              'weights-----------------------')
        class_weights = list(class_weight.compute_class_weight('balanced',
                                                               np.unique(
                                                                   y_train),
                                                               y_train))
        print(class_weights)

        return x_train, x_test, y_train, y_test, x_train_vectorized, \
               x_test_vectorized, tf_idf_vectorizer

