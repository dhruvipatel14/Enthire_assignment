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
        df['airline_sentiment'].value_counts().iplot(kind='bar',asFigure=True)

    def freq_count(self,x):
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
        Visualization.show_wordcolud(self,df)
        Visualization.bar_plot(self,df)
        Visualization.freq_count(self,df.text)
