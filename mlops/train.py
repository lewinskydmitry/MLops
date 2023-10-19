import pandas as pd

from mlops.models.basic_model import Baseline_classifier
from mlops.tools.train_model import train_model


class Trainer:
    def create_model(self, num_features, num_parameters):
        model = Baseline_classifier(num_features, num_parameters)
        return model

    def load_data(self, path):
        df = pd.read_csv(path)
        return df

    def train_model(
        self, path_to_data, batch_size=256, num_parameters=256, num_epoch=10
    ):
        df = self.load_data(path_to_data)
        num_features = len(df) - 1
        model = self.create_model(num_features, num_parameters)
        train_model(df, num_epoch, batch_size, model)
