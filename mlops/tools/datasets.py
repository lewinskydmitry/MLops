import numpy as np
import torch
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
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


class InferDataset(Dataset):
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
        return x