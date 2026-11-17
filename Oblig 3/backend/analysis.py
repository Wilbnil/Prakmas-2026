#File made for dumping the analysis results from the notebook to be used in the backend

import json
import os

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, TunedThresholdClassifierCV


dataset = pd.read_csv('../Dataset/weatherAUS.csv')
dataset = dataset.loc[
    (dataset['Rainfall'] < 350)
    & (dataset['Evaporation'] < 100)
    & (dataset['WindSpeed9am'] < 100)
].dropna(subset=['RainTomorrow'])

features = dataset.drop(columns=['RainTomorrow'])
targets = dataset['RainTomorrow'].map({'Yes': 1, 'No': 0})

models = {
    'Random Forest': joblib.load('models/rf_pipe2_fold_1.pkl'),
    'Gradient Boosting': joblib.load('models/gb_pipe_fold_1.pkl'),
    'Multilayer Perceptron': joblib.load('models/nnMLP_pipe_fold_1.pkl'),
}

splitter = TimeSeriesSplit(n_splits=5, gap=1)
dates = pd.Series(dataset['Date'])
unique_dates = pd.Series(dataset['Date'].unique())

train_indices, test_indices = next(splitter.split(unique_dates))
train_rows = dates.isin(unique_dates.iloc[train_indices])
test_rows = dates.isin(unique_dates.iloc[test_indices])

features_train = features.loc[train_rows]
targets_train = targets.loc[train_rows]
features_test = features.loc[test_rows]
targets_test = targets.loc[test_rows]

os.makedirs('results', exist_ok=True)
keys = {
    'Random Forest': 'rf',
    'Gradient Boosting': 'gb',
    'Multilayer Perceptron': 'mlp',
}
test_scores = {'y_true': targets_test.astype(int).tolist(), 'models': {}}
default_thresholds = {}
importance = {}

for name, grid in models.items():
    key = keys[name]
    probabilities = grid.predict_proba(features_test)[:, 1]
    test_scores['models'][key] = [round(float(p), 4) for p in probabilities]

    tuned = TunedThresholdClassifierCV(
        grid.best_estimator_,
        cv=5,
        scoring=make_scorer(fbeta_score, beta=2),
    )
    tuned.fit(features_train, targets_train)
    default_thresholds[key] = float(tuned.best_threshold_)
    print(name, 'best threshold:', tuned.best_threshold_)

    result = permutation_importance(
        grid.best_estimator_,
        features_test,
        targets_test,
        scoring='roc_auc',
        n_repeats=5,
        random_state=0,
        n_jobs=-1,
    )
    #Makes a list of dictionaries containing each feature and its mean and standard deviation for the specific model (key)
    importance[key] = sorted(
        [
            {'feature': feature, 'mean': float(mean), 'std': float(std)}
            for feature, mean, std in zip(
                features_test.columns,
                result.importances_mean,
                result.importances_std,
            )
        ],
        key=lambda item: -item['mean'], #Sorted from highest to lowest mean so the higher importance features are first
    )

with open('results/test_scores.json', 'w') as file:
    json.dump(test_scores, file)
with open('results/default_thresholds.json', 'w') as file:
    json.dump(default_thresholds, file)
with open('results/importance.json', 'w') as file:
    json.dump(importance, file)

print('Saved results to results/')