import fire

from mlops.infer import Inferencer
from mlops.train import Trainer


def train(
    path_to_data: str,
    batch_size: int = 256,
    num_parameters: int = 256,
    num_epoch: int = 10,
) -> None:
    """
    Train a model.

    Args:
        path_to_data (str): Path to the training data.
        batch_size (int): Batch size for training (default: 256).
        num_parameters (int): Number of model parameters (default: 256).
        num_epoch (int): Number of training epochs (default: 10).
    """
    Trainer().train_model(path_to_data, batch_size, num_parameters, num_epoch)


def infer(batch_size: int = 256) -> None:
    """
    Make inferences using a trained model.

    Args:
        batch_size (int): Batch size for making inferences (default: 256).
    """
    Inferencer().make_infer(batch_size)


if __name__ == "__main__":
    fire.Fire({"train": train, "infer": infer})
