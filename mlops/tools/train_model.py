import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from .datasets import TrainingDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(df, num_epoch, batch_size, model):

    params_opt = dict(lr=1e-4)
    opt = torch.optim.AdamW(model.parameters(), **params_opt)

    X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Machine failure']),
                                                 df['Machine failure'],
                                                 shuffle=True,
                                                 stratify=df['Machine failure'], random_state=42,
                                                 train_size=0.7)
    
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis = 1)

    train_dataset = TrainingDataset(df_train)
    val_dataset = TrainingDataset(df_test)

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=True)

    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size, 
        shuffle=True)

    for epoch in range(num_epoch):
        model.train(True)
        for X_batch, y_batch in train_dl:
                opt.zero_grad()

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                pred = torch.softmax(logits, dim=1)

                train_loss = nn.CrossEntropyLoss()(pred, y_batch)
                train_loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                pred = torch.softmax(logits, dim=1)
                test_loss = nn.CrossEntropyLoss()(pred, y_batch)
        print(f'train loss = {train_loss}, test loss = {test_loss}')

    if not os.path.exists('mlops/saved_models'):
        os.makedirs('mlops/saved_models')
    torch.save(model.state_dict(), 'mlops/saved_models/classifier_model.pth')