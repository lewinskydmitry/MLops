from pathlib import Path

import fire
import yaml

from mlops.infer import Inferencer
from mlops.train import Trainer


def load_config(config_file):
    with open(config_file, "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)


def train() -> None:
    """
    Function for training the model. To change the parameters of training please modify the config.yaml
    """
    config = load_config("mlops/config.yaml")
    file_path = Path(config["path"])

    if file_path.exists():
        Trainer().train_model(
            config["path"],
            config["train_model"]["batch_size"],
            config["train_model"]["num_parameters"],
            config["train_model"]["num_epoch"],
        )
    else:
        raise FileNotFoundError("Load data using DVC")


def infer() -> None:
    """
    Make inferences using a trained model. To change the parameters of inferencing please modify the config.yaml
    """
    config = load_config("./config.yaml")
    file_path = Path(config["path"])

    if file_path.exists():
        Inferencer().make_infer(config["infer_model"]["batch_size"])
    else:
        raise FileNotFoundError(
            "Please run 'python commands.py train' to train model first"
        )


if __name__ == "__main__":
    fire.Fire({"train": train, "infer": infer})
