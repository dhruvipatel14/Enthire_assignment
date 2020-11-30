from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config, notfound_view_config
import json
import pyramid
import pyramid_swagger
from Preprocessing import read_data,preprocess_data,data_visualization
from Training import train_test_vectorized, model
from Predictions import model_prediction
import pandas as pd
pd.set_option('max_colwidth', 800)
pd.set_option('display.max_columns', 5)
pd.set_option('display.width', 2000)
import cufflinks as cf
cf.go_offline()
from plotly.offline import iplot
import plotly.graph_objs as go
import pickle
pyramid.includes = pyramid_swagger

class Runall:

    def operations(self):
        df = read_data.ReadData.read_dataset(self)
        preprocess_df = preprocess_data.Preprocess.preprocessed_text(self,df=df)
        print(preprocess_df.head())
        data_visualization.Visualization.all_plots(self,preprocess_df)
        x_train, x_test, y_train, y_test, x_train_vectorized, \
        x_test_vectorized  = \
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

@notfound_view_config()
def notfound(request):
    return Response('Response not found', status='404 not found')


@view_config(route_name='predict', renderer='json', request_method='POST')
def get_predict(request):
    request_data = request.json_body['text']

    tfidf_obj = pickle.load(open('tf_idf.pkl','rb'))
    model = pickle.load(open('logistic.pkl', 'rb'))
    test_vectorized = tfidf_obj.transform([request_data])
    predictions = (model.predict(test_vectorized)).tolist()

    response = dict()
    response['sentiment'] = "".join(predictions)
    return Response(json.dumps(response))

if __name__ == '__main__':

    runall = Runall()
    runall.operations()

    settings = {'pyramid_swagger.schema_directory': 'api_docs/',
                'pyramid_swagger.enable_swagger_spec_validation':
                    'pyramid_debugtoolbar pyramid_tm',
                'pyramid_swagger.enable_request_validation ': True,
                'pyramid_swagger.enable_response_validation': True,
                'pyramid_swagger.include_missing_properties': True,
                'pyramid_swagger.enable_api_doc_views': True
                }

    config = Configurator(settings=settings)
    config.include('pyramid_swagger')
    config.add_static_view('static', 'static', cache_max_age=3600)

    config.add_route('predict', '/predict')
    config.add_view(get_predict, route_name='predict')

    app = config.make_wsgi_app()
    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()

