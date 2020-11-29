from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config, notfound_view_config
import json
from json import loads
from Preprocessing import read_data,preprocess_data,data_visualization
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
        data_visualization.Visualization.all_plots(self,preprocess_df)

if __name__ == '__main__':
    runall = Runall()
    runall.operations()
