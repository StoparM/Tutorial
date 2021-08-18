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

    # from sklearn.ensemble import RandomForestClassifier
    # model = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state = 0).fit(x_train, y_train)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver='sag', random_state=0).fit(x_train, y_train)

    train_score = model.score(x_train, y_train) * 100
    print(train_score)

    test_score = model.score(x_test, y_test) * 100
    print(test_score)

    predicted_val = model.predict(x_test)

    precision, recall, prc_thresholds = metrics.precision_recall_curve(y_test, predicted_val)
    fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, predicted_val)

    avg_prec = metrics.average_precision_score(y_test, predicted_val)
    roc_auc = metrics.roc_auc_score(y_test, predicted_val)

    scores_file = config["reports"]["scores"]
    prc_file = config["reports"]["prc"]
    roc_file = config["reports"]["roc"]
    auc_file = config["reports"]["auc"]

    nth_point = math.ceil(len(prc_thresholds)/1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]    
    
    with open(prc_file, "w") as fd:
        prcs = {
                "prc": [
                    {"precision": p, "recall": r, "threshold": t}
                    for p, r, t in prc_points
                ]
            }
        json.dump(prcs, fd, indent=3, cls=NumpyEncoder)
        

    with open(roc_file, "w") as fd:
        rocs = {
                "roc": [
                    {"fpr": fp, "tpr": tp, "threshold": t}
                    for fp, tp, t in zip(fpr, tpr, roc_thresholds)
                ]
            }
        json.dump(rocs, fd, indent=4, cls=NumpyEncoder)
        

    print(classification_report(y_test, predicted_val))

    # cm = confusion_matrix(y_test, predicted_val)
    # print(cm)

        
    # df1 = pd.DataFrame(predicted_val, columns = ['Predicted'])
    # df_cm = pd.concat([y_test, df1], axis=1)
    # print(df_cm)
        
    # df_cm.to_csv('cm.csv', index = False)

    # roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    # print('ROC_AUC:{0:0.2f}'.format(roc_auc))

    Logistic_Accuracy = accuracy_score(y_test, predicted_val)
    print('Logistic Regression Model Accuracy:{0:0.2f}'.format(Logistic_Accuracy))

    average_precision = average_precision_score(y_test, predicted_val)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    with open(scores_file, "w") as f:
        scores = {
            "train_score": train_score,
            "test_score": test_score,
 #           "roc_auc": roc_auc,
            "Logistic Accuracy": Logistic_Accuracy      
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