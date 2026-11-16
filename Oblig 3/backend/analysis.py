# Import required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('default')
import seaborn as sns
import json, os

# Save and load GridSearchCV objects:
from os import path as pth
import joblib


#Sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, confusion_matrix, \
    ConfusionMatrixDisplay, roc_curve, auc, PrecisionRecallDisplay, fbeta_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, TunedThresholdClassifierCV
from sklearn.inspection import permutation_importance

# Feature importance
import shap as sh

from encoding import *

# Read the dataset
dataset_AUS = pd.read_csv('../Dataset/weatherAUS.csv')
# Uses existing column names baked into csv file.

# Constant for target value
TARGET = "RainTomorrow"

#Outlier removal based on EDA
dataset_AUS = dataset_AUS.loc[(dataset_AUS['Rainfall'] < 350)
                                        & (dataset_AUS['Evaporation'] < 100)
                                        & (dataset_AUS['WindSpeed9am'] < 100)]
# Creating a temporary dataset
dataset_AUS_temp = dataset_AUS.copy()

# Removing rows where RainTomorrow is missing
dataset_AUS_temp = dataset_AUS_temp.dropna(subset=['RainTomorrow'])

#Converting the Date feature to datetime for sorting
dataset_AUS_temp['Date'] = pd.to_datetime(dataset_AUS_temp['Date'])

#Sorting the dataset chronologically
dataset_AUS_temp = dataset_AUS_temp.sort_values(by=['Date'])

# Temporary separating into targets and features
targets = dataset_AUS_temp['RainTomorrow'].map({'Yes' : 1, 'No' : 0})
features = dataset_AUS_temp.drop(columns=['RainTomorrow'])

# Select numerical columns
numerical_features = features.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

# Select categorical columns
categorical_features = features.select_dtypes(
    include=['object', 'category', 'string']
).columns.tolist()

# Numerical pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Location pipeline
location_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', TargetEncoder(target_type='binary',))
])

rain_today_pipeline = Pipeline([
    ('encoder', FunctionTransformer(encode_rain_today)),
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Wind pipeline
wind_cyclical_pipeline = Pipeline([
    ('encoder', FunctionTransformer(encode_wind)),
    ('imputer', SimpleImputer(strategy='constant'))
])

# Date pipeline
date_cyclical_pipeline = Pipeline([
    ('encoder', FunctionTransformer(encode_date)),
    ('imputer', SimpleImputer(strategy='median'))
])

# Preprocessor
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('location', location_pipeline, location_features),
    ('rain_today', rain_today_pipeline, rain_today_features),
    ('wind', wind_cyclical_pipeline, wind_features),
    ('date', date_cyclical_pipeline, date_features)
])

# Function for saving and loading GridSearchCV objects
def getModelGrid(pipeline, params,features_train, targets_train, inner_splits, path="Models/default_model.pkl", overwrite_existing=0, n_jobs = 1, score = "recall"):
    # Fit model if either the flag for a new model is active or no model is found. If the flag is false and a model is found, the old model is loaded.
    if not pth.isfile(path) or overwrite_existing:
        grid = GridSearchCV(estimator=pipeline, param_grid=params, cv=inner_splits, n_jobs=n_jobs, scoring=score)
        grid.fit(features_train, targets_train)
        joblib.dump(grid, path)
        print(f"Grid saved at: {path}")
    else:
        grid = joblib.load(path)
        print(f"Grid loaded from: {path}")
    return grid, pd.DataFrame(grid.cv_results_)

def nested_cross_validation(
        model_pipeline,
        param_grid,
        features=features,
        targets=targets,
        outer_split=5,
        inner_split=3,
        gap = 1,
        n_jobs = 1,
        score='recall',
        save_path='Models',
        model_name = 'default',
        overwrite_existing = 0
):

    dates = pd.Series(dataset_AUS_temp['Date'])
    unique_dates = pd.Series(dataset_AUS_temp['Date'].unique())
    #Creating outer and inner CV loop
    outer_cv = TimeSeriesSplit(
        n_splits=outer_split,
        gap=gap
    )
    inner_cv = TimeSeriesSplit(
        n_splits=inner_split,
        gap=gap
        )
    assert dates.index.equals(features.index) and dates.index.equals(targets.index)
    #Storing results
    outer_results = []
    best_models = []
    best_params = []
    grid_results = []

    #Outer CV iterates on unique dates so that there won't be any mixing of testing and validation on the same day
    for fold, (trainCvIndices, testIndices) in enumerate(outer_cv.split(unique_dates), 1):

        #Saves the training+validation and test dates
        outer_train_cv_dates = unique_dates.iloc[trainCvIndices]
        outer_test_dates = unique_dates.iloc[testIndices]

        #Checking which rows belong to Training+Validation and Testing
        outer_train_cv_rows = dates.isin(outer_train_cv_dates)
        outer_test_rows = dates.isin(outer_test_dates)

        #Distributing the features and targets for CV and testing
        xTrainCv = features.loc[outer_train_cv_rows]
        yTrainCv = targets.loc[outer_train_cv_rows]

        xTest = features.loc[outer_test_rows]
        yTest = targets.loc[outer_test_rows]

        #Inner CV dates
        inner_train_cv_rows = dates.loc[outer_train_cv_rows]

        inner_splits = []
        #Inner CV splits the outer CV training dates into training and validation groups
        for inner_train_index, inner_val_index in inner_cv.split(outer_train_cv_dates):
            inner_train_dates = outer_train_cv_dates.iloc[inner_train_index]
            inner_val_dates = outer_train_cv_dates[inner_val_index]

            #Makes lists of indices that are to be included in the training and validation splits for GridSearchCV
            inner_train_rows = np.flatnonzero(
                inner_train_cv_rows.isin(
                    inner_train_dates).to_numpy()
            )
            inner_val_rows = np.flatnonzero(
                inner_train_cv_rows.isin(
                    inner_val_dates).to_numpy()
            )

            inner_splits.append((inner_train_rows, inner_val_rows))

        grid, grid_result = getModelGrid(
            pipeline=model_pipeline,
            params=param_grid,
            features_train=xTrainCv,
            targets_train=yTrainCv,
            inner_splits=inner_splits,
            path=f"{save_path}/{model_name}_fold_{fold}.pkl",
            overwrite_existing=overwrite_existing,
            n_jobs=n_jobs,
            score=score,
        )

        grid_results.append(grid_result)

        print(
                f"Fold {fold}: "
                f"Training: {outer_train_cv_dates.min()} to {outer_train_cv_dates.max()}, "
                f"Testing: {outer_test_dates.min()} to {outer_test_dates.max()}"
        )

        #Finding the best performing models and their parameters
        predictions = grid.best_estimator_.predict(xTest)

        #Evaluation
        results = {
            "Fold": fold,
            "Accuracy": accuracy_score(yTest, predictions),
            "Precision": precision_score(yTest, predictions, zero_division=0),
            "Recall": recall_score(yTest, predictions, zero_division=0),
            "f1": f1_score(yTest, predictions, zero_division=0),
        }

        outer_results.append(results)
        best_models.append(grid.best_estimator_)
        best_params.append(grid.best_params_)

    return outer_results, best_models, best_params, grid_results

def printResults(results_list):
    for results in results_list:
        for key, value in results.items():
            print(f"{key}: {value:.3f}")

#%%
# Start by fetching the models from their saved location.
# Manually load saved models:

path = "models/rf_pipe2_fold_1.pkl"
rf_grid = joblib.load(path)

path = "models/gb_pipe_fold_1.pkl"
gb_grid = joblib.load(path)


path = "models/nnMLP_pipe_fold_1.pkl"
nnMLP_grid = joblib.load(path)

# Dictionary of models to use in evaluation
models = {"Random Forest": rf_grid,
          "Gradient Boosting": gb_grid,
          "Multilayer Perceptron": nnMLP_grid}

#%%
# Function for returning average results between folds.
# After other changes in the notebook, this is only used once.
def getAverageMetrics(results):
    # Initialize lists of metric values for current fold
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    # Partition every fold's metrics into each metric.
    for fold in results:
        accuracy_list.append(fold["Accuracy"])
        precision_list.append(fold["Precision"])
        recall_list.append(fold["Recall"])
        f1_list.append(fold["f1"])

        # Average each metric across folds.
    accuracy = np.average(accuracy_list)
    precision = np.average(precision_list)
    recall = np.average(recall_list)
    f1 = np.average(f1_list)

    return accuracy, precision, recall, f1

# The table is based on the last assignment's result table.

# Function which takes in a model name, its result and the table to append to.
def metrics_table_append(model_name, results, table):

    # Fetch average results between folds for the given model's results.
    accuracy, precision, recall, f1 = getAverageMetrics(results)

    row={
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
    }

    # Returns the dataframe passed in through the table parameter as a dataframe with an additional row
    # of metrics for one model.
    return pd.concat([table, pd.DataFrame([row])], ignore_index=True)

model_names = list(models.keys())

# A function for plotting a confusion matrix for a model on the entire dataset.
def show_confusion(model, ax, title):
    predictions = model.predict(features)
    confusion = confusion_matrix(targets, predictions)
    display = ConfusionMatrixDisplay(confusion)
    display.plot(ax=ax)
    ax.set_title(title)

def show_roc(models):
    # Baseline: The ROC value for randomly guessing target.
    plt.plot([0,1], [0,1], linestyle='--')
    # Ground truth
    y_true = targets
    for model_key, model in models.items():
        # The models' probability of choosing target values.
        # Initially passed with both positive and negative probabilities.
        # roc_curve only takes positive, hence the [:,1].
        probabilities = model.predict_proba(features)
        fpr, tpr, _= roc_curve(y_true, probabilities[:,1])
        plt.plot(fpr, tpr, label=model_key)
        # Prints the area under the ROC curve for all models.
        print(f"Area Under Curve for {model_key} = {auc(fpr, tpr):.3f}")

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

# Start by creating test features and targets for fold 1 to match the model's fold.

# Duplicate dataset for use on threshold tuning.
dataset_AUS_temp = dataset_AUS.copy()

# Delete rows where target is missing:
dataset_AUS_temp.dropna(subset=['RainTomorrow'], inplace=True)

# Separate features and targets.
features = dataset_AUS_temp.drop(columns=['RainTomorrow'])
targets = dataset_AUS_temp['RainTomorrow'].map({'Yes' : 1, 'No' : 0})


# Only if preprocessor needs to be applied:
# Ensure preprocessor pipeline returns pandas DataFrames.
# preprocessor.set_output(transform='pandas')
# Use preprocessor pipeline on the duplicated dataset.
# Returns only features since target is simply mapped earlier.
# features_processed = preprocessor.fit_transform(features, targets)

dates = pd.Series(dataset_AUS_temp['Date'])
unique_dates = pd.Series(dataset_AUS_temp['Date'].unique())
# Recreating outer loop
outer_test = TimeSeriesSplit(
    n_splits=5,
    gap=1
)
# Initialize dictionaries to manually access trainval and test from fold 1.
xTrainval_dict = {}
yTrainval_dict = {}
xTest_dict = {}
yTest_dict = {}
for fold, (trainCvIndices, testIndices) in enumerate(outer_test.split(unique_dates), 1):


        # Save trainval dates.
        outer_trainval_dates = unique_dates.iloc[trainCvIndices]

        # Checking which rows belong to Trainval.
        outer_trainval_rows = dates.isin(outer_trainval_dates)

        xTrainVal = features.loc[outer_trainval_rows]
        yTrainVal = targets.loc[outer_trainval_rows]
        xTrainval_dict.update({fold : xTrainVal})
        yTrainval_dict.update({fold : yTrainVal})

        # Saves the test dates.
        outer_test_dates = unique_dates.iloc[testIndices]

        # Checking which rows belong to Testing.
        outer_test_rows = dates.isin(outer_test_dates)

        xTest = features.loc[outer_test_rows]
        yTest = targets.loc[outer_test_rows]
        xTest_dict.update({fold : xTest})
        yTest_dict.update({fold : yTest})



# Trainval dataframes for fold 1.
featuresTrainVal_fold1 = xTrainval_dict[1]
targetsTrainVal_fold1 = yTrainval_dict[1]

# Testing dataframes for fold 1.
featuresTest_fold1 = xTest_dict[1]
targetsTest_fold1 = yTest_dict[1]

# Test Function used after tuning.
def test(model):
    train_predictions = model.predict(featuresTest_fold1)
    print(f'Test Accuracy: {accuracy_score(targetsTest_fold1, train_predictions,):.2f}')
    print(f'Test \033[1mPrecision: {precision_score(targetsTest_fold1, train_predictions):.2f}\033[0m')
    print(f'Test \033[1mRecall: {recall_score(targetsTest_fold1, train_predictions):.2f}\033[0m')
    print(f'Test F1: {f1_score(targetsTest_fold1, train_predictions):.2f}')
    print(f'Test Fbeta: {fbeta_score(targetsTest_fold1, train_predictions, beta=beta):.2f}')

# The beta weighting value which is used for the fbeta score.
# Higher beta
beta = 2

rf_tuned = TunedThresholdClassifierCV(rf_grid, cv=5, scoring=make_scorer(fbeta_score, beta=beta))
rf_tuned.fit(featuresTrainVal_fold1, targetsTrainVal_fold1)

# Test the threshold tuning
test(rf_tuned)

os.makedirs("results", exist_ok=True)
KEYS = {"Random Forest": "rf", "Gradient Boosting": "gb", "Multilayer Perceptron": "mlp"}

test_scores = {"y_true": targetsTest_fold1.astype(int).tolist(), "models": {}}
default_thresholds, importance = {}, {}

for display_name, grid in models.items():
    key = KEYS[display_name]

    # 1) Test-set probabilities (backend recomputes metrics/curves at any threshold from these)
    proba = grid.predict_proba(featuresTest_fold1)[:, 1]
    test_scores["models"][key] = [round(float(p), 4) for p in proba]

    # 2) Threshold tuning (F-beta, beta=2) - tuned on the best pipeline, trainval data only
    tuned = TunedThresholdClassifierCV(grid.best_estimator_, cv=5,
                                       scoring=make_scorer(fbeta_score, beta=beta))
    tuned.fit(featuresTrainVal_fold1, targetsTrainVal_fold1)
    default_thresholds[key] = float(tuned.best_threshold_)
    print(display_name, "best threshold:", tuned.best_threshold_)

    # 3) Feature importance: permutation importance on the raw (un-preprocessed) columns
    r = permutation_importance(grid.best_estimator_, featuresTest_fold1, targetsTest_fold1,
                               scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=-1)
    importance[key] = sorted(
        [{"feature": f, "mean": float(m), "std": float(s)}
         for f, m, s in zip(featuresTest_fold1.columns, r.importances_mean, r.importances_std)],
        key=lambda d: -d["mean"])

json.dump(test_scores, open("results/test_scores.json", "w"))
json.dump(default_thresholds, open("results/default_thresholds.json", "w"))
json.dump(importance, open("results/importance.json", "w"))
print("Saved results to backend/results/")