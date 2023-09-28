import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class Baseline_classifier(nn.Module):
    def __init__(self, num_features, init_param):
        super(Baseline_classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(num_features, init_param),
            nn.BatchNorm1d(init_param),
            nn.ReLU(),
            nn.Linear(init_param, int(init_param/32)),
            nn.BatchNorm1d(int(init_param/32)),
            nn.ReLU(),
            nn.Linear(int(init_param/32), 2))

    def forward(self, x):
        x = self.classifier(x)
        return x


def train_eval(df, num_epoch, batch_size, model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params_opt = dict(lr=1e-4)
    opt = torch.optim.AdamW(model.parameters(), **params_opt)


    X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Machine failure']),
                                                 df['Machine failure'],
                                                 shuffle=True,
                                                 stratify=df['Machine failure'], random_state=42,
                                                 train_size=0.7)
    
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis = 1)

    train_dataset = ClassifierDataset(df_train)
    val_dataset = ClassifierDataset(df_test)

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
                train_loss = nn.CrossEntropyLoss()(logits, y_batch)
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


class ClassifierDataset(Dataset):
    def __init__(self, data):
        self.data = np.array(data)
        self.features = self.data[:, :-1]
        self.labels = self.data[:, -1]

        mean = np.mean(self.features, axis=0)
        std = np.std(self.features, axis=0)

        self.features = (self.features - mean) / std
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y