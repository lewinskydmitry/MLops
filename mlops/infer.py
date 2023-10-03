import pandas as pd
import torch
from models.basic_model import Baseline_classifier

from mlops.tools.inference_model import infer_model

NUM_FEATURES = 9
BATCH_SIZE = 512
NUM_PARAMETERS = 256


def main():
    df = pd.read_csv("mlops/data/prepared_data.csv")
    model = Baseline_classifier(NUM_FEATURES, NUM_PARAMETERS)
    model.load_state_dict(torch.load("mlops/saved_models/classifier_model.pth"))

    model.eval()
    infer_model(df, BATCH_SIZE, model)


if __name__ == "__main__":
    main()
