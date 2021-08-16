import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params

def split_and_saved_data(config_path):

    config = read_params(config_path)
    x_test_data_path = config["split_data"]["x_test_path"] 
    x_train_data_path = config["split_data"]["x_train_path"]
    y_test_data_path = config["split_data"]["y_test_path"] 
    y_train_data_path = config["split_data"]["y_train_path"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]

    df = pd.read_csv(raw_data_path, header=None)
    x = df.iloc[:, 0:300].values
    y = df.iloc[:, 300:301].values

    x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=split_ratio, random_state=random_state)
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    x_test = pd.DataFrame(x_test)
    y_test = pd.DataFrame(y_test)

    x_train.to_csv(x_train_data_path, header=None, index=None)
    y_train.to_csv(y_train_data_path, header=None, index=None)
    x_test.to_csv(x_test_data_path, header=None, index=None)
    y_test.to_csv(y_test_data_path, header=None, index=None)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)