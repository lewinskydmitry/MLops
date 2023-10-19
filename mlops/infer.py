import pandas as pd
import torch
from models.basic_model import Baseline_classifier

from mlops.tools.inference_model import infer_model

BATCH_SIZE = 512


def main():
    df = pd.read_csv("mlops/data/prepared_data.csv")
    parameters = torch.load("mlops/saved_models/classifier_model.pth")
    model = Baseline_classifier(
        parameters["classifier.0.weight"].shape[1],
        parameters["classifier.0.weight"].shape[0],
    )
    model.load_state_dict(parameters)

    model.eval()
    infer_model(df, BATCH_SIZE, model)


if __name__ == "__main__":
    main()
