import pandas as pd
from models.basic_model import Baseline_classifier
from tools.train_model import train_model

NUM_FEATURES = 9
NUM_PARAMETERS = 256
NUM_EPOCH = 10
BATCH_SIZE = 512


def main():
    df = pd.read_csv("mlops/data/prepared_data.csv")
    model = Baseline_classifier(NUM_FEATURES, NUM_PARAMETERS)
    train_model(df, NUM_EPOCH, BATCH_SIZE, model)


if __name__ == "__main__":
    main()
