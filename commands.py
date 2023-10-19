import fire

from mlops.infer import Inferencer
from mlops.train import Trainer


def train(
    path_to_data: str,
    batch_size: int = 256,
    num_parameters: int = 256,
    num_epoch: int = 10,
) -> None:
    Trainer().train_model(path_to_data, batch_size, num_parameters, num_epoch)


def infer(batch_size: int = 256) -> None:
    Inferencer().make_infer(batch_size)


if __name__ == "__main__":
    fire.Fire()
