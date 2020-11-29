from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config, notfound_view_config
import json
from json import loads
from Preprocessing import read_data,preprocess_data,data_visualization
from Training import train_test_vectorized, model
from Predictions import model_prediction
import pandas as pd
pd.set_option('max_colwidth', 800)
# pd.options.display.float_format = "{:.2f}".format
pd.set_option('display.max_columns', 5)
pd.set_option('display.width', 2000)
import cufflinks as cf
cf.go_offline()
from plotly.offline import iplot
import plotly.graph_objs as go

class Runall:

    def operations(self):
        df = read_data.ReadData.read_dataset(self)
        preprocess_df = preprocess_data.Preprocess.preprocessed_text(self,df=df)
        print(preprocess_df.head())
        # data_visualization.Visualization.all_plots(self,preprocess_df)
        x_train, x_test, y_train, y_test, x_train_vectorized, \
        x_test_vectorized, tf_idf_vectorizer = \
            train_test_vectorized.Pretraining(
            preprocess_df).tfidf()

        model.Train_model(x_train_vectorized,y_train).random_forest()
        model.Train_model(x_train_vectorized, y_train).logistic_regression()
        model.Train_model(x_train_vectorized, y_train).xgboost_classifier()

        print('-------------------model prediction and '
              'accuracy--------------------')
        print('-------------------------------Random '
              'Forest-----------------------')

        model_prediction.MakePrediction(x_test_vectorized,
                                        y_test).predications(
            'rf_classifier.pkl')

        print('-------------------------------Xgboost-----------------------')

        model_prediction.MakePrediction(x_test_vectorized,
                                        y_test).predications(
            'xgboost.pkl')
        print('-------------------------------Logistic '
              'Regression-----------------------')

        model_prediction.MakePrediction(x_test_vectorized,
                                        y_test).predications(
            'logistic.pkl')

if __name__ == '__main__':
    runall = Runall()
    runall.operations()
