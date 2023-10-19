import fire
import pandas as pd
import torch
from models.basic_model import Baseline_classifier

from mlops.tools.inference_model import infer_model


class Inferencer:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def make_infer(self):
        df = pd.read_csv("mlops/data/prepared_data.csv")
        parameters = torch.load("mlops/saved_models/classifier_model.pth")
        model = Baseline_classifier(
            parameters["classifier.0.weight"].shape[1],
            parameters["classifier.0.weight"].shape[0],
        )
        model.load_state_dict(parameters)

        model.eval()
        infer_model(df, self.batch_size, model)


if __name__ == "__main__":
    fire.Fire(Inferencer)
