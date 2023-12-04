import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch import seed_everything
from .datasets import TrainingDataset

seed_everything(42, workers=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_to_model = 'mlops/saved_models'


def train_model(df, num_epoch, batch_size, model):

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["Machine failure"]),
        df["Machine failure"],
        shuffle=True,
        stratify=df["Machine failure"],
        random_state=42,
        train_size=0.7,
    )

    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    train_dataset = TrainingDataset(df_train)
    val_dataset = TrainingDataset(df_test)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    trainer = L.Trainer(max_epochs=num_epoch, default_root_dir=path_to_model)
    trainer.fit(model=model, train_dataloaders=train_dl)
    trainer.validate(model=model, dataloaders=val_dl)