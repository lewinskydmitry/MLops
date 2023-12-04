import pandas as pd

from mlops.models.lightning_model import Lightning_classifier
from mlops.tools.train_model import train_model


class Trainer:
    def create_model(
        self, num_features: int, num_parameters: int
    ) -> Lightning_classifier:
        """
        Create a baseline classifier model.

        Args:
            num_features (int): Number of features in the input data.
            num_parameters (int): Number of parameters for the model.

        Returns:
            Baseline_classifier: An instance of the baseline classifier model.
        """
        model = Lightning_classifier(num_features, num_parameters)
        return model

    def load_data(self, path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded DataFrame from the CSV file.
        """
        df = pd.read_csv(path)
        return df

    def train_model(
        self,
        path_to_data: str,
        batch_size: int = 256,
        num_parameters: int = 256,
        num_epoch: int = 10,
    ) -> None:
        """
        Train a baseline classifier model using the provided data.

        Args:
            path_to_data (str): Path to the CSV file containing the training data.
            batch_size (int, optional): Batch size for training (default: 256).
            num_parameters (int, optional): Number of parameters for the model (default: 256).
            num_epoch (int, optional): Number of training epochs (default: 10).
        """
        df = self.load_data(path_to_data)
        num_features = len(df.columns) - 1
        model = self.create_model(num_features, num_parameters)
        train_model(df, num_epoch, batch_size, model)
