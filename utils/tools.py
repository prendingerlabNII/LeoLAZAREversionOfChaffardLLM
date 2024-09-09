import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
from datetime import datetime
from distutils.util import strtobool
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema

from utils.metrics import metric

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    # if args.decay_fac is None:
    #     args.decay_fac = 0.5
    # if args.lradj == 'type1':
    #     lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    # elif args.lradj == 'type2':
    #     lr_adjust = {
    #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #         10: 5e-7, 15: 1e-7, 20: 5e-8
    #     }
    if args.lradj =='type1':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj =='type2':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    elif args.lradj =='type4':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch) // 1))}
    else:
        args.learning_rate = 1e-4
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    print("lr_adjust = {}".format(lr_adjust))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:

                print(line)

                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def vali(model, vali_data, vali_loader, criterion, args, device, itr):
    total_loss = []
    
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


    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
        model.eval()
    else:
        model.in_layer.eval()
        model.out_layer.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            sup = batch_y[:, -args.pred_len:, -1].to(device)
            res = batch_y[:, -args.pred_len:, -1].to(device)

            outputs = model(batch_x, itr)
                
            # encoder - decoder
            outputs = outputs[:, -args.pred_len:, -4:]
            batch_y = batch_y[:, -args.pred_len:, -4:].to(device)
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            sup = sup.detach().cpu()
            res = res.detach().cpu()
            
            # if args.pct :
            #     pred = torch.cumprod(pred+1, dim = 0)
            #     true = torch.cumprod(true+1, dim = 0)

        if args.loss_func == 'wmse':
            loss = criterion(pred, true, sup, res)
        else:
            loss = criterion(pred, true)

        total_loss.append(loss)
    total_loss = np.average(total_loss)
    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
        model.train()
    else:
        model.in_layer.train()
        model.out_layer.train()
    return total_loss

def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))

def sup_res_lines(price):
    indicesbot = argrelextrema(price, comparator = np.less, order = 7)[0]
    indicestop = argrelextrema(price, comparator = np.greater, order = 7)[0]
    ext = np.concatenate([indicestop,indicesbot])
    km = KMeans(n_clusters= 3).fit(price[ext].reshape((-1,1)))
    return km.cluster_centers_

def test(model, test_data, test_loader, args, device, itr):
    preds = []
    trues = []
    inputx = []
    # mases = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            # outputs_np = batch_x.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_input_itr{}_{}.npy".format(itr, i), outputs_np)
            # outputs_np = batch_y.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_true_itr{}_{}.npy".format(itr, i), outputs_np)

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            outputs = model(batch_x[:, -args.seq_len:, :], 0)
            
            # encoder - decoder
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            input = [batch_x[i].detach().cpu().numpy() for i in range(len(batch_x))]
            if (test_data.scale):
                input = [test_data.inverse_transform(batch_x[i].detach().cpu().numpy()) for i in range(len(batch_x))  ]
                pred = [test_data.inverse_transform(pred[i]) for i in range(len(pred))]
                true = [test_data.inverse_transform(true[i]) for i in range(len(true))]
            input = np.array(input)
            pred = np.array(pred)
            true = np.array(true)
            
            preds.append(pred)
            trues.append(true)
            inputx.append(input)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)
    # mases = np.mean(np.array(mases))
    res = {'preds': preds, 'trues' : trues, 'inputx': inputx }
    print('test shape:', preds.shape, trues.shape)
    if args.scale :
        with open(f'./datasets/results/{args.model}_{args.data_path}_voc{args.voc}_seq{args.seq_len}_patch{args.patch_size}_pred{args.pred_len}_stride{args.stride}_itr{itr}.pkl','wb') as f :
            pkl.dump(res, f)
    else :
        with open(f'./datasets/results/{args.model}_{args.data_path}_voc{args.voc}_seq{args.seq_len}_patch{args.patch_size}_pred{args.pred_len}_stride{args.stride}_itr{itr}.pkl','wb') as f :
            pkl.dump(res, f)
    print('done')
    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    print('rmse:{:.4f}, smape:{:.4f}'.format(rmse, smape))

    return preds, trues, inputx, rmse, smape

def test_1(model, test_data, test_loader, args, device, itr):
    preds = []
    trues = []
    trues_mark = []
    # mases = []
    scaler = test_data.scaler
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            
            # outputs_np = batch_x.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_input_itr{}_{}.npy".format(itr, i), outputs_np)
            # outputs_np = batch_y.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_true_itr{}_{}.npy".format(itr, i), outputs_np)

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            # print(batch_x.shape)
            # print(batch_y.shape)
            outputs = model(batch_x[:, -args.seq_len:, :], itr)
            
            # encoder - decoder
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            # pred = scaler.inverse_transform(pred)
            # true = scaler.inverse_transform(true)
            preds.append(pred)
            trues.append(true)
    preds = np.array(preds)
    trues = np.array(trues)
    trues = trues.reshape(546,256,7)
    preds= preds.reshape(546,256,7)
    # mases = np.mean(np.array(mases))
    print('test shape:', preds.shape, trues.shape)
    # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    # preds = preds.reshape((len(preds),7))
    # trues = trues.reshape((len(trues),7))
    print(scaler.mean_)
    print(scaler.scale_)
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    # print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}, mases:{:.4f}'.format(mae, mse, rmse, smape, mases))
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    return preds, trues, mse, mae

def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # compute sum of differences between line and prices, 
    # return negative val if invalid 
    
    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
     
    diffs = line_vals - y
    
    # Check to see if the line is valid, return -1 if it is not valid.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line 
    err = (diffs ** 2.0).sum()
    return err


def optimize_slope(support: bool, pivot:int , init_slope: float, y: np.array):
    
    # Amount to change slope by. Multiplyed by opt_step
    slope_unit = (y.max() - y.min()) / len(y) 
    
    # Optmization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step # current step
    
    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert(best_err >= 0.0) # Shouldn't ever fail with initial slope

    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount
            # to see if error increases/decreases. 
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err
            
            # If increasing by a small amount fails, 
            # try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0: # Derivative failed, give up
                raise Exception("Derivative failed. Check your data. ")

            get_derivative = False

        if derivative > 0.0: # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else: # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step
        

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err: 
            # slope failed/didn't reduce error
            curr_step *= 0.5 # Reduce step size
        else: # test slope reduced error
            best_err = test_err 
            best_slope = test_slope
            get_derivative = True # Recompute derivative
    
    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y[pivot])


def fit_trendlines_single(data: np.array):
    # find line of best fit (least squared) 
    # coefs[0] = slope,  coefs[1] = intercept 
    t = np.arange(len(data))
    coefs = np.polyfit(t, data, 1)

    # Get points of line.
    line_points = coefs[0] * t + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax() 
    lower_pivot = (data - line_points).argmin() 
    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)
    
    return (support_coefs, resist_coefs) , line_points

def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    # coefs[0] = slope,  coefs[1] = intercept
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax() 
    lower_pivot = (low - line_points).argmin() 
    
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)