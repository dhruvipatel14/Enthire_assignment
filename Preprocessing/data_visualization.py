from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)
import cufflinks as cf
cf.go_offline()
import plotly
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.graph_objs as go
import pandas as pd
import nltk
import seaborn as sns

class Visualization:
    def show_wordcolud(self,data):
        """
        Generates word cloud for given corpus
        :param data: corpus
        :return: shows plot
        """
        wordcloud = WordCloud(
            background_color='black',
            stopwords=stopwords,
            max_font_size=40,
            scale=3,
            random_state=1  # chosen at random by flipping a coin; it was heads
        ).generate(str(data))

        fig = plt.figure(1, figsize=(15, 15))
        plt.imshow(wordcloud)
        plt.show()

    def bar_plot(self,df):
        """
        generates bar plot for airline sentiments , which shows ratio of
        positve and negative sentiments in data
        :param df: dataframe of given dataset
        :return:
        """
        df['airline_sentiment'].value_counts().iplot(kind='bar',asFigure=True)

    def freq_count(self,x):
        """
        Count the frequency of words in given courpus and displays top 30
        words in corpus
        :param x: series of text data
        :return:
        """
        all_words = ' '.join([text for text in x])
        all_words = all_words.split()
        fdist = nltk.FreqDist(all_words)
        words_df = pd.DataFrame(
            {'word': list(fdist.keys()), 'count': list(fdist.values())})

        d = words_df.nlargest(columns="count", n=30)

        plt.figure(figsize=(12, 15))
        ax = sns.barplot(data=d, x="count", y="word")
        ax.set(ylabel='Word')
        plt.show()

    def all_plots(self,df):
        """
        Calls all the graphs of text analysis
        :param df: Dataframe
        :return:
        """
        Visualization.show_wordcolud(self,df)
        Visualization.bar_plot(self,df)
        Visualization.freq_count(self,df.text)
