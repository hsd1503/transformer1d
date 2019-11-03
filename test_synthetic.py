"""
test on synthetic data

Shenda Hong, Nov 2019
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report 

from util import read_data_generated
from transformer1d import Transformer1d, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

    
if __name__ == "__main__":
    
    # make data
    n_samples = 1000
    n_length = 50
    n_channel = 16
    n_classes = 3    
    data, label = read_data_generated(n_samples=n_samples, n_length=n_length, n_channel=n_channel, n_classes=n_classes)
    print(data.shape, Counter(label))
    dataset = MyDataset(data, label)
    dataloader = DataLoader(dataset, batch_size=64)
    
    # make model
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    ## change the hyper-parameters for your own data
    model = Transformer1d(
        n_classes=n_classes, 
        n_length=n_length, 
        d_model=n_channel, 
        nhead=8, 
        dim_feedforward=128, 
        dropout=0.1, 
        activation='relu')
    model.to(device)

    # train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()
    all_loss = []
    prog_iter = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(prog_iter):

        input_x, input_y = tuple(t.to(device) for t in batch)
        input_x = input_x.permute(2,0,1)
        pred = model(input_x)

        loss = loss_func(pred, input_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        all_loss.append(loss.item())
    
    plt.plot(all_loss)
    
    # test
    data_test, label_test = read_data_generated(n_samples=n_samples, n_length=n_length, n_channel=n_channel, n_classes=n_classes)
    print(data_test.shape, Counter(label_test))
    dataset_test = MyDataset(data_test, label_test)
    dataloader_test = DataLoader(dataset_test, batch_size=64, drop_last=False)
    prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
    all_pred_prob = []
    for batch_idx, batch in enumerate(prog_iter_test):
        input_x, input_y = tuple(t.to(device) for t in batch)
        input_x = input_x.permute(2,0,1)
        pred = model(input_x)
        all_pred_prob.append(pred.cpu().data.numpy())
    all_pred_prob = np.concatenate(all_pred_prob)
    all_pred = np.argmax(all_pred_prob, axis=1)
    ## classification report
    print(classification_report(all_pred, label_test))
    
    
    
    