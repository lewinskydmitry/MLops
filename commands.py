import csv
import os
from pathlib import Path

import dvc.api
import fire
import yaml

from mlops.infer import Inferencer
from mlops.train import Trainer


def load_config(config_file):
    with open(config_file, "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)


config = load_config("mlops/config.yaml")
data_path = config["path"]


def pull_data():
    # Use the `dvc.api.read` function to download and read the data
    with dvc.api.open(
        data_path, repo="https://github.com/lewinskydmitry/mlops", mode="r"
    ) as file:
        data = file.read()

    # Assuming 'data' is a string representing CSV data
    # You may need to adjust this based on the actual structure of your data
    num_features = len(data.split("\n")[0].split(","))
    csv_data = [row.split(",") for row in data.split("\n") if len(row) > num_features]

    # Creating folders
    directory_path = os.path.dirname(data_path)
    os.makedirs(directory_path, exist_ok=True)
    csv_file_path = os.path.join(directory_path, "data.csv")

    # Use the `csv` module to write the data to a CSV file
    with open(csv_file_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        for row in csv_data:
            csv_writer.writerow(row)

    print(f"CSV file saved at {data_path}")


def train() -> None:
    """
    Function for training the model. To change the parameters of training please modify the config.yaml
    """

    if ~Path(data_path).exists():
        pull_data()

    Trainer().train_model(
        config["path"],
        config["train_model"]["batch_size"],
        config["train_model"]["num_parameters"],
        config["train_model"]["num_epoch"],
    )


def infer() -> None:
    """
    Make inferences using a trained model. To change the parameters of inferencing please modify the config.yaml
    """

    if Path(data_path).exists():
        Inferencer().make_infer(data_path, config["infer_model"]["batch_size"])
    else:
        raise FileNotFoundError(
            "Please run 'python commands.py train' to train model first"
        )


if __name__ == "__main__":
    fire.Fire({"train": train, "infer": infer})
