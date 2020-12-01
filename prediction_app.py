from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config, notfound_view_config
import json
import pyramid
import pyramid_swagger

import pickle
pyramid.includes = pyramid_swagger

@notfound_view_config()
def notfound(request):
    return Response('Response not found', status='404 not found')


@view_config(route_name='predict', renderer='json', request_method='POST')
def get_predict(request):
    """
    Predict the sentiment of given text
    :param request: contains params from request API
    :return: response with predicted sentiment
    """
    request_data = request.json_body['text']

    tfidf_obj = pickle.load(open('tf_idf.pkl','rb'))
    model = pickle.load(open('logistic.pkl', 'rb'))
    test_vectorized = tfidf_obj.transform([request_data])
    predictions = (model.predict(test_vectorized)).tolist()

    response = dict()
    response['sentiment'] = "".join(predictions)
    return Response(json.dumps(response))


if __name__ == '__main__':

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

