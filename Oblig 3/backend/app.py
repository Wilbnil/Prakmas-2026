import json, os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             fbeta_score, confusion_matrix, roc_curve, precision_recall_curve,
                             roc_auc_score)
import __main__, encoding
for _n in dir(encoding):
    if _n.startswith('encode_') or _n == 'cyclic_encoder':
        setattr(__main__, _n, getattr(encoding, _n))

app = Flask('My Backend')
CORS(app)
api = Api(app)

FEATURES = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir',
            'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
            'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']
CATEGORICAL = {'Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'}

NAMES = {'mlp': 'Multilayer Perceptron', 'gb': 'Gradient Boosting', 'rf': 'Random Forest'}
MODELS = {
    'mlp': joblib.load('models/nnMLP_pipe_fold_1.pkl'),
    'gb': joblib.load('models/gb_pipe_fold_1.pkl'),
    'rf': joblib.load('models/rf_pipe2_fold_1.pkl'),
}

def load_json(path, default):
    return json.load(open(path)) if os.path.isfile(path) else default

DEFAULTS = load_json('results/default_thresholds.json', {})   # tuned thresholds from analysis
THRESHOLDS = {n: float(DEFAULTS.get(n, 0.5)) for n in MODELS}
SCORES = load_json('results/test_scores.json', None)           # fold-1 test probabilities
IMPORTANCE = load_json('results/importance.json', {})

def unknown(name):
    return {'error': f'unknown model {name}'}, 404

def to_frame(data):
    row = {}
    for f in FEATURES:
        v = data.get(f)
        if v is None or v == '':
            row[f] = np.nan
        elif f in CATEGORICAL:
            row[f] = v
        else:
            row[f] = float(v)
    df = pd.DataFrame([row], columns=FEATURES)
    # Keep categorical columns as object dtype (as in training), even when blank
    df[list(CATEGORICAL)] = df[list(CATEGORICAL)].astype(object)
    return df

def thin(xs, ys, n=200):
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    return [{'x': round(float(xs[i]), 4), 'y': round(float(ys[i]), 4)} for i in idx]

class ModelList(Resource):
    def get(self):
        return [{'id': k, 'name': NAMES[k], 'threshold': THRESHOLDS[k],
                 'default_threshold': float(DEFAULTS.get(k, 0.5))} for k in MODELS]

class Predict(Resource):
    def post(self, name):
        if name not in MODELS:
            return unknown(name)
        inputs = to_frame(request.get_json(force=True) or {})
        proba = float(MODELS[name].predict_proba(inputs)[0, 1])
        t = THRESHOLDS[name]
        return {'proba': proba, 'label': int(proba >= t), 'threshold': t}

class Threshold(Resource):
    def get(self, name):
        if name not in MODELS:
            return unknown(name)
        return {'threshold': THRESHOLDS[name]}

    def put(self, name):
        if name not in MODELS:
            return unknown(name)
        value = (request.get_json(force=True) or {}).get('threshold')
        try:
            value = float(value)
        except (TypeError, ValueError):
            return {'error': 'threshold must be a number'}, 400
        if not 0 <= value <= 1:
            return {'error': 'threshold must be between 0 and 1'}, 400
        THRESHOLDS[name] = value
        return {'threshold': value}

class Metrics(Resource):
    def get(self, name):
        if name not in MODELS:
            return unknown(name)
        if not SCORES:
            return {'error': 'results/test_scores.json missing - run the export code in analysis.py'}, 503
        y = np.array(SCORES['y_true'])
        p = np.array(SCORES['models'][name])
        t = float(request.args.get('threshold', THRESHOLDS[name]))
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        sweep = []
        for th in np.round(np.arange(0, 1.0001, 0.02), 2):
            q = (p >= th).astype(int)
            sweep.append({'threshold': float(th),
                          'precision': precision_score(y, q, zero_division=0),
                          'recall': recall_score(y, q, zero_division=0),
                          'f1': f1_score(y, q, zero_division=0),
                          'f2': fbeta_score(y, q, beta=2, zero_division=0)})
        return {
            'threshold': t,
            'metrics': {'accuracy': accuracy_score(y, pred),
                        'precision': precision_score(y, pred, zero_division=0),
                        'recall': recall_score(y, pred, zero_division=0),
                        'f1': f1_score(y, pred, zero_division=0),
                        'f2': fbeta_score(y, pred, beta=2, zero_division=0),
                        'auc': roc_auc_score(y, p)},
            'confusion': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'roc': thin(fpr, tpr),
            'pr': thin(rec, prec),
            'operating_point': {'fpr': float(fp / max(fp + tn, 1)), 'tpr': float(tp / max(tp + fn, 1))},
            'sweep': sweep,
        }

class Importance(Resource):
    def get(self, name):
        if name not in MODELS:
            return unknown(name)
        return {'method': 'Permutation importance (drop in ROC AUC on the fold-1 test set)',
                'features': IMPORTANCE.get(name, [])}

api.add_resource(ModelList, '/models')
api.add_resource(Predict, '/models/<string:name>/predict')
api.add_resource(Threshold, '/models/<string:name>/threshold')
api.add_resource(Metrics, '/models/<string:name>/metrics')
api.add_resource(Importance, '/models/<string:name>/importance')

if __name__ == '__main__':
    app.run(debug=True)
