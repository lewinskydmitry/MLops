import argparse
from mlops.models.models import *

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
        help='batch_size'
    )

    args = parser.parse_args()
    return args.model


# def main(args: argparse.Namespace):
#     model = Baseline_classifier(256,9)
#     train_eval(args.epoch, args.batch_size, model)

def main(epoch,batch_size):
    df = pd.read_csv('mlops/data/prepared_data.csv')
    model = Baseline_classifier(9, 256)
    train_eval(df, epoch, batch_size, model)

    

if __name__ == '__main__':
    # arguments = parse_args()
    main(40,256)