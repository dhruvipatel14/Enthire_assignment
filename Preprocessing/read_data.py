import pandas as pd
pd.set_option('max_colwidth', 800)


class ReadData:
    def __init__(self):
        """

        """
    def read_dataset(self):
        data = pd.read_csv(
            r'F:\online_competitions\airline_sentiment_analysis.csv')
        df = pd.DataFrame(data)
        print(df.head())
        print("shape of data",df.shape)
        df.rename(columns={"Unnamed: 0": "id"}, inplace=True)
        return df
