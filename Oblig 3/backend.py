import joblib
import numpy as np
import pickle as p
import pandas as pd
from flask import Flask
from flask_restful import Resource, Api
from flask_restful.reqparse import RequestParser

app = Flask('My Backend')
api = Api(app)

def cyclic_encoder(series, period):
    sin = np.sin(2 * np.pi * series / period)
    cos = np.cos(2 * np.pi * series / period)

    return sin, cos

location_features = ['Location']

rain_today_features = ['RainToday']

wind_features = [
    'WindGustDir',
    'WindDir9am',
    'WindDir3pm'
]

date_features = ['Date']
wind_map = {
    'N': 1, 'NNE': 2, 'NE': 3, 'ENE': 4,
    'E': 5, 'ESE': 6, 'SE': 7, 'SSE': 8,
    'S': 9, 'SSW': 10, 'SW': 11, 'WSW': 12,
    'W': 13, 'WNW': 14, 'NW': 15, 'NNW': 16
}

WIND_DIRECTIONS = 16
MONTH_PERIOD = 12

def encode_rain_today(rain_today_dataframe):
    return rain_today_dataframe.replace({'No': 0, 'Yes': 1}).astype(float)

def encode_wind(wind_data_frame):
    result = pd.DataFrame()

    for column in wind_data_frame.columns:

        wind_numbers = wind_data_frame[column].map(wind_map).astype(float)

        sin, cos = cyclic_encoder(wind_numbers, WIND_DIRECTIONS)
        result[column + '_Sine'], result[column + '_Cosine'] = sin, cos

    return result

def encode_date(dataframe):
    result = pd.DataFrame()

    date = pd.to_datetime(dataframe['Date'])
    year = date.dt.year
    month = date.dt.month
    day = date.dt.day_of_year

    is_leap_year = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    days_in_year = np.select([is_leap_year], [366], default = 365)

    result['Month_Sine'], result['Month_Cosine'] = cyclic_encoder(month, MONTH_PERIOD)

    result['Day_Sine'], result['Day_Cosine'] = cyclic_encoder(day, days_in_year)

    result['Year'] = year

    return result

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