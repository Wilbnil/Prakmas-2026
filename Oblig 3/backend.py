import joblib
from flask import Flask
from flask_restful import Resource, Api
from flask_restful.reqparse import RequestParser
from encoding import *

app = Flask('My Backend')
api = Api(app)

feature_names = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir',
                 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']
MODELS = {
    'mlp': joblib.load('models/nnMLP_pipe_fold_1.pkl'),
    'gb': joblib.load('models/gb_pipe_fold_1.pkl'),
    'rf': joblib.load('models/rf_pipe2_fold_1.pkl'),
}
THRESHOLDS = {name: 0.5 for name in MODELS}

class Predict(Resource):
    def __init__(self):
        self.reqparser = RequestParser()
        for name in feature_names:
            self.reqparser.add_argument(name, required=True, location='args')

    def get(self, name):
        if name not in MODELS:
            return {'error': f"unknown model {name}" }, 404
        args = self.reqparser.parse_args()
        inputs = pd.DataFrame([args], columns=feature_names)

        proba = float(MODELS[name].predict_proba(inputs)[0,1])
        threshold = THRESHOLDS[name]
        label = int (proba >= threshold)
        print(MODELS[name].classes_)

        return {'proba': proba, 'label': label, 'threshold': threshold}

class Threshold(Resource):
    def get(self, name):
        if name not in MODELS:
            return {'error': f"unknown model {name}" }, 404
        return {'threshold': THRESHOLDS[name]}

    def put(self, name):
        if name not in MODELS:
            return {'error': f"unknown model {name}" }, 404
        parser = RequestParser()
        parser.add_argument('threshold', required=True, type=float, location='json')
        args = parser.parse_args()

        if not (0 <= args['threshold'] <= 1):
            return {'error': f"threshold must be between 0 and 1" }, 400

        THRESHOLDS[name] = args['threshold']

        return THRESHOLDS[name]

api.add_resource(Predict, '/models/<string:name>/predict')
api.add_resource(Threshold, '/models/<string:name>/threshold')

if __name__ == '__main__':
    app.run(debug=True)