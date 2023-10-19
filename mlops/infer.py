import pandas as pd
import torch

from mlops.models.basic_model import Baseline_classifier
from mlops.tools.inference_model import infer_model


class Inferencer:
    def make_infer(self, batch_size: int = 256):
        """
        Perform inference using a baseline classifier model on prepared data.

        Args:
            batch_size (int, optional): Batch size for inference (default: 256).
        """
        # Load the prepared data from a CSV file
        df = pd.read_csv("mlops/data/prepared_data.csv")

        # Load the saved model parameters from a PyTorch file
        parameters = torch.load("mlops/saved_models/classifier_model.pth")

        # Create an instance of the Baseline_classifier model using the loaded parameters
        model = Baseline_classifier(
            parameters["classifier.0.weight"].shape[1],
            parameters["classifier.0.weight"].shape[0],
        )
        model.load_state_dict(parameters)

        # Set the model to evaluation mode
        model.eval()

        # Perform inference using the loaded data, batch size, and model
        infer_model(df, batch_size, model)
