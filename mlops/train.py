import fire
import pandas as pd
from models.basic_model import Baseline_classifier
from tools.train_model import train_model


class Trainer:
    def __init__(self, num_epoch):
        self.num_epoch = num_epoch

    def create_model(self, num_features, num_parameters):
        model = Baseline_classifier(num_features, num_parameters)
        return model

    def load_data(self, path):
        df = pd.read_csv(path)
        return df

    def train_model(self, path_to_data, batch_size, num_parameters):
        df = self.load_data(path_to_data)
        num_features = len(df) - 1
        model = self.create_model(num_features, num_parameters)
        train_model(df, self.num_epoch, batch_size, model)


if __name__ == "__main__":
    fire.Fire(Trainer)
