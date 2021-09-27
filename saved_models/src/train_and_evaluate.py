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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import average_precision_score
from numpyencoder import NumpyEncoder
from get_data import read_params
import matplotlib.pyplot as plt
import argparse
import joblib
import json
from sklearn.metrics import plot_confusion_matrix

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()



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
    model = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state = 999).fit(x_train, y_train.ravel())
    
    # DecisionTreeClassifier(criterion = 'entropy', random_state = 999)
    # LogisticRegression()
    # GaussianNB()
    # RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state = 999)


    predicted_val = model.predict(x_test)

    anomalies = ('NiAnomalije', 'InstaD', 'SlowD', 'SuddenR', 'SuddenD', )
    cm = confusion_matrix(y_test, predicted_val, normalize='true')
    disp = plot_confusion_matrix(cm, target_names=anomalies)
    plt.savefig('confusion_matrix.png')


    scores_file = config["reports"]["scores"]

    Logistic_Accuracy = accuracy_score(y_test, predicted_val)
    print('Accuracy:{0:0.2f}'.format(Logistic_Accuracy))

    with open(scores_file, "w") as f:
        scores = {
            "Accuracy": Logistic_Accuracy      
        }
        json.dump(scores, f)


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(model, model_path)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
