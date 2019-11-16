"""
test on physionet data

Shenda Hong, Nov 2019
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from util import read_data_physionet_2, read_data_physionet_4, preprocess_physionet
from transformer1d import Transformer1d, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

    
if __name__ == "__main__":

    is_debug = False
    
    batch_size = 64
    if is_debug:
        writer = SummaryWriter('/nethome/shong375/log/transformer1d/challenge2017/debug')
    else:
        writer = SummaryWriter('/nethome/shong375/log/transformer1d/challenge2017/first')

    # make data
    # preprocess_physionet() ## run this if you have no preprocessed data yet
    window_size = 1000
    X_train, X_test, Y_train, Y_test, pid_test = read_data_physionet_4(window_size=window_size)
    print(X_train.shape)
    dataset = MyDataset(X_train, Y_train)
    dataset_test = MyDataset(X_test, Y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False)
    
    # make model
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    kernel_size = 16
    stride = 2
    n_block = 16
    model = Transformer1d(
        n_classes=4, 
        n_length=window_size, 
        d_model=1, 
        nhead=1, 
        dim_feedforward=128, 
        dropout=0.1, 
        activation='relu',
        verbose = True)
    model.to(device)
    ## look model
    prog_iter = tqdm(dataloader, desc="init", leave=False)
    for batch_idx, batch in enumerate(prog_iter):
        input_x, input_y = tuple(t.to(device) for t in batch)
        pred = model(input_x)    
        break

    # train and test
    model.verbose = False
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss()

    n_epoch = 50
    step = 0
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):

        # train
        model.train()
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):

            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            writer.add_scalar('Loss/train', loss.item(), step)

            if is_debug:
                break
        
        scheduler.step(_)
                    
        # test
        model.eval()
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        all_pred_prob = []
        for batch_idx, batch in enumerate(prog_iter_test):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            all_pred_prob.append(pred.cpu().data.numpy())
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)
        ## vote most common
        final_pred = []
        final_gt = []
        for i_pid in np.unique(pid_test):
            tmp_pred = all_pred[pid_test==i_pid]
            tmp_gt = Y_test[pid_test==i_pid]
            final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
            final_gt.append(Counter(tmp_gt).most_common(1)[0][0])
        ## classification report
        tmp_report = classification_report(final_gt, final_pred, output_dict=True)
        print(confusion_matrix(final_gt, final_pred))
        f1_score = (tmp_report['0']['f1-score'] + tmp_report['1']['f1-score'] + tmp_report['2']['f1-score'] + tmp_report['3']['f1-score'])/4
        writer.add_scalar('F1/f1_score', f1_score, _)
        writer.add_scalar('F1/label_0', tmp_report['0']['f1-score'], _)
        writer.add_scalar('F1/label_1', tmp_report['1']['f1-score'], _)
        writer.add_scalar('F1/label_2', tmp_report['2']['f1-score'], _)
        writer.add_scalar('F1/label_3', tmp_report['3']['f1-score'], _)

    
    
    
    