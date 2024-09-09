
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from models.GPT4TS import GPT4TS
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Custom
import pickle as pkl
import os
from einops import rearrange
plt.rcParams['savefig.facecolor'] = "0.8"
device = torch.device('cuda:0')

def visual(data, lookback, index, pred_len, showSMA = True, showEMA = True, showMSE = True, savefig = False):
    """
    Results visualization
    """
    with open(f'./datasets/results/{data}.pkl', 'rb') as file :
        res = pkl.load(file)
    preds = res ['preds'] 
    trues = res ['trues'] 
    inputx = res['inputx']
    plt.figure(dpi = 400)
    plt.xlabel('Time (hours))')
    plt.ylabel('Price (USD)')
    
    truth = np.concatenate((inputx[index,-lookback:,-1], trues[index,:,-1]), axis = 0)
    pred = np.concatenate((inputx[index, -lookback:, -1], preds[index, :, -1]), axis=0)
    opens = np.concatenate((inputx[index, -lookback:, 0], trues[index, :, 0]), axis=0)
    high = np.concatenate((inputx[index, -lookback:, 1], trues[index, :, 1]), axis=0)
    low = np.concatenate((inputx[index, -lookback:, 2], trues[index, :, 2]), axis=0)
    if showMSE :
        Mse = []
        for i in range(pred_len):
            Mse.append(np.sqrt(np.mean(np.square(trues[:,i,-1]-preds[:,i,-1]))))
    print(Mse)
    if showEMA:
        EMA50 = np.concatenate((inputx[index, -lookback:, 6], trues[index, :, 6]), axis=0)
    # if showSMA:
    #     SMA24= np.concatenate((inputx[index, -lookback:, 5], trues[index, :, 5]), axis=0)
    height = truth - opens
    bottom = np.where(height > 0, opens, truth + abs(height))
    color = np.where(height > 0, 'g', 'r')
    # plt.plot(EMA50, label = '50 hour EMA', color = 'lightblue', linestyle = '--')
    # plt.plot(SMA24, label = '24 hour SMA', color = 'lightseagreen', linestyle = '--')
    plt.plot(range(lookback-1,lookback +pred_len),pred[-pred_len-1:], label = 'prediction', color = 'purple', marker = '+')
    print(pred.shape)
    plt.fill_between(range(lookback,lookback +pred_len),pred[-pred_len:]+Mse,pred[-pred_len:]-Mse,alpha = 0.2, color = 'g')
    plt.bar(range(len(truth)), height, bottom=bottom, color=color, align='center')
    plt.vlines(range(len(high)), ymin=low, ymax=high, color=color, linewidth=1)
    plt.legend()
    plt.plot()
    if savefig :
        plt.savefig(f'./pic/daily/{data}_index{index}.png', dpi = 400)

def RMSE(data, pred_len):

    with open(f'./datasets/results/{data}.pkl', 'rb') as file :
        res = pkl.load(file)
    preds = res ['preds'] 
    trues = res ['trues'] 
    inputx = res['inputx']
    Mse = []
    for i in range(pred_len):
        Mse.append(np.sqrt(np.mean(np.square(trues[:,i,-1]-preds[:,i,-1]))))
    Rmse = np.sqrt(np.mean(np.square(trues[:,:,-1]-preds[:,:,-1])))
    return Mse, Rmse