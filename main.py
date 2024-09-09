from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, vali, test
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.LLaMa2_old import Llama
from models.LLaMa2_test_backup import Llamatest
from models.LLaMa2_Daily import LlamaDaily
from models.TimeLLM import Model
from models.Yi import Yi
from huggingface_hub import login

import numpy as np
import torch
import torch.nn as nn
from torch import optim

CUDA_LAUNCH_BLOCKING=1.

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--num_features', type=int, default=4)

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--scale', type=int, default = 0)
parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float,default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--isllama', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float,default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)
parser.add_argument('--voc', type=int, default = 15 )
parser.add_argument('--pct', type = int, default = 1)
parser.add_argument('--pca', type = int, default = 1)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)



args = parser.parse_args()

SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "to_torch.bfloat16hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

mses = []
maes = []
rmses = []
smapes = []

login(token='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')

for ii in range(args.itr):

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, args.seq_len, args.label_len, args.pred_len,
                                                                    args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                    args.d_ff, args.embed, 0)
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.freq == 0:
        args.freq = 'h'

    if args.scale == 0 :
        args.scale = False
    elif args.scale == 1 :
        args.scale == True
    
    if args.pct == 0 :
        args.pct = False
    elif args.pct == 1 :
        args.pct == True

    if args.isllama == 1:
        args.isllama = True

    elif args.isllama == 0:
        args.isllama = False

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    if args.freq != 'h':
        args.freq = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(args.freq))

    device = torch.device('cuda:0')

    time_now = time.time()
    train_steps = len(train_loader)

    if args.model == 'PatchTST':
        model = PatchTST(args, device)
        model.to(device)
    elif args.model == 'DLinear':
        model = DLinear(args, device)
        model.to(device)
    elif args.model == 'GPT4TS':
        model = GPT4TS(args, device)
    elif args.model == 'llamatest':
        model = Llamatest(args, device)
    elif args.model == 'llamadaily':
        model = LlamaDaily(args, device)
    elif args.model == 'Yi':
        model = Yi(args, device)
    elif args.model == 'timellm':
        model = Model(args, device)
    else :
        model = Llama(args, device)
    # mse, mae = test(model, test_data, test_loader, args, device, ii)

    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    if args.loss_func == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_func == 'wmse':
        class wMSE(nn.Module):
            def _init_(self):
                super(wMSE, self).__init__()
            def forward(self, pred, true, sup, res):
                mse = (pred - true) ** 2
                weights = torch.maximum(sup,res)
                weights = weights.unsqueeze(2)
                weights = torch.broadcast_to(weights, (5,7,4))
                res = torch.mean(mse*weights.expand_as(true))
                return res
        criterion = wMSE()

    elif args.loss_func == 'smape':
        class SMAPE(nn.Module):
            def __init__(self):
                super(SMAPE, self).__init__()
            def forward(self, pred, true):
                return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
        criterion = SMAPE()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            sup = batch_y[:, -args.pred_len:, -1].to(device)
            res = batch_y[:, -args.pred_len:, -1].to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            # print(batch_x.shape)
            with torch.autocast(device_type= "cuda", dtype = torch.bfloat16):
                outputs = model(batch_x, ii)
            
                # ouputs = model(batch_x,batch_x_mark, dec_inp, batch_y_mark)
                
                outputs = outputs[:, -args.pred_len:,-4:]
                batch_y = batch_y[:, -args.pred_len:, -4:].to(device)
                if args.pct :
                    outputs = torch.cumprod(outputs+1, dim = 1)
                    batch_y = torch.cumprod(batch_y+1, dim = 1)

                # print(outputs)
                if args.loss_func == 'wmse':
                    loss = criterion(outputs, batch_y, sup, res )
                else:
                    loss = criterion(outputs, batch_y)
                # print(loss)
                train_loss.append(loss.item())

            if (i + 1) % 1000 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            loss.backward()
            model_optim.step()

        
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
        # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
        # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
        #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))

        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_model_path = path + '/' + 'checkpoint.pth'
    model.eval()
    model.load_state_dict(torch.load(best_model_path))
    print("------------------------------------")
    preds, trues, inputx, rmse, smape = test(model, test_data, test_loader, args, device, 0)
    rmses.append(rmse)
    smapes.append(smape)

rmses = np.array(rmses)
smapes = np.array(smapes)

print("rmse_mean = {:.4f}, rmse_std = {:.4f}".format(np.mean(rmses), np.std(rmses)))
print("smape_mean = {:.4f}, smape_std = {:.4f}".format(np.mean(smapes), np.std(smapes)))