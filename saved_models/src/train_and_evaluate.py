import os
import math
import warnings
import sys
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, plot_roc_curve,accuracy_score
from sklearn.metrics import average_precision_score
from numpyencoder import NumpyEncoder
from get_data import read_params
import matplotlib.pyplot as plt
import argparse
import joblib
import json
from sklearn.metrics import plot_confusion_matrix

#https://www.analyticsvidhya.com/blog/2021/06/mlops-tracking-ml-experiments-with-data-version-control/

def train_and_evaluate(config_path):

    config = read_params(config_path)
    x_test_data_path = config["split_data"]["x_test_path"] 
    x_train_data_path = config["split_data"]["x_train_path"]
    y_test_data_path = config["split_data"]["y_test_path"] 
    y_train_data_path = config["split_data"]["y_train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    target = [config["base"]["target_col"]]

    x_train = pd.read_csv(x_train_data_path, header=None)
    x_test = pd.read_csv(x_train_data_path, header=None)
    y_train = pd.read_csv(y_train_data_path, header=None)
    y_test = pd.read_csv(y_train_data_path, header=None)

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver='sag', random_state=0).fit(x_train, y_train.ravel())


    disp = plot_confusion_matrix(model, x_test, y_test, normalize='true',cmap=plt.cm.Blues)
    plt.savefig('confusion_matrix.png')

    train_score = model.score(x_train, y_train) * 100
    test_score = model.score(x_test, y_test) * 100
    predicted_val = model.predict(x_test)

    scores_file = config["reports"]["scores"]

    Logistic_Accuracy = accuracy_score(y_test, predicted_val)
    print('Accuracy:{0:0.2f}'.format(Logistic_Accuracy))

    with open(scores_file, "w") as f:
        scores = {
            "Accuracy": Logistic_Accuracy      
        }
        json.dump(scores, f, indent=3)


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(model, model_path)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)