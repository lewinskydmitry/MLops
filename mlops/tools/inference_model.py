import os
import torch
import numpy as np
import torch.nn as nn

from .datasets import InferDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infer_model(infer_data, batch_size, model):

    infer_dataset = InferDataset(infer_data)

    infer_dl = DataLoader(
        infer_dataset,
        batch_size=batch_size, 
        shuffle=True)
    
    predicts = torch.tensor([])
    with torch.no_grad():
        for X_batch in infer_dl:
            X_batch = X_batch.to(device)
            
            logits = model(X_batch)
            pred = torch.softmax(logits, dim=1)
            predicts = torch.cat([predicts,pred],axis = 0)
    
    if not os.path.exists('mlops/predictions'):
        os.makedirs('mlops/predictions')
    np.savetxt('mlops/predictions/predictions.csv', predicts.numpy(), delimiter=',')
    