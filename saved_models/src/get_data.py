import os
import yaml
import pandas as pd
import argparse

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    data_path = config["data_source"]["source"]

    df = pd.read_csv(data_path, header=None)
    df = df.replace('NiAnomalije', 0)
    df = df.replace('InstaD', 1)
    df = df.replace('SlowD', 2)
    df = df.replace('SuddenR', 3)
    df = df.replace('SuddenD', 4)

    return df

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path = parsed_args.config)