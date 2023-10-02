
import torch
import pandas as pd
import numpy as np
import argparse

from models.basic_model import Baseline_classifier
from tools.eval_model import eval_model



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Training and evaluation parameters'
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
    model.load_state_dict(torch.load('mlops/saved_models/classifier_model.pth'))
    model.eval()

    eval_model(df, 300, model)


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)