import pandas as pd
import torch

from mlops.models.basic_model import Baseline_classifier
from mlops.tools.inference_model import infer_model


class Inferencer:
    def make_infer(self, batch_size: int = 256):
        df = pd.read_csv("mlops/data/prepared_data.csv")
        parameters = torch.load("mlops/saved_models/classifier_model.pth")
        model = Baseline_classifier(
            parameters["classifier.0.weight"].shape[1],
            parameters["classifier.0.weight"].shape[0],
        )
        model.load_state_dict(parameters)

        model.eval()
        infer_model(df, batch_size, model)
