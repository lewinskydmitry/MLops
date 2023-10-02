import pandas as pd
import argparse
from tools.train_model import train_model
from models.basic_model import Baseline_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Training and evaluation parameters'
    )

    parser.add_argument(
        '--epoch', type=int, required=True,
        help='Num epoch for training'
    )

    parser.add_argument(
        '--batch_size', type=int, required=False,
        help='batch size'
    )

    args = parser.parse_args()

    return args


def main(args: argparse.Namespace):
    df = pd.read_csv('mlops/data/prepared_data.csv')
    model = Baseline_classifier(9, 256)
    train_model(df, args.epoch, args.batch_size, model)


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
    