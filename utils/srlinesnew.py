import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # compute sum of differences between line and prices, 
    # return negative val if invalid 
    
    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y.iloc[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
     
    diffs = line_vals - y
    
    # Check to see if the line is valid, return -1 if it is not valid.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line 
    err = (diffs ** 2.0).sum()
    # err = np.sum(np.abs(diffs))
    return err;


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
            derivative = test_err - best_err;
            
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
    return (best_slope, -best_slope * pivot + y.iloc[pivot])


def adjust_slope(pivot:int , init_slope: float, y: np.array):
    # Coordonnées du point de pivot
    xp = pivot
    yp = y.iloc[pivot]
    
    min_diff_slope = np.inf
    for i in range(len(y)):
        if i == pivot:
            continue
        # Calcul de la pente entre ce point et le point de pivot
        slope = (y.iloc[i] - yp) / (i - xp)
        diff_slope = abs(slope - init_slope)
        if diff_slope < min_diff_slope:
            min_diff_slope = diff_slope
            min_index = i
    
    # Calcul de la nouvelle pente et de l'intercept
    if min_index is not None:
        new_slope = (y.iloc[min_index] - yp) / (min_index - xp)
        new_intercept = yp - new_slope * xp
        return (new_slope, new_intercept)
    else:
        return (None, None)
    

def readjust_line(init_slope: float, init_intercept: float, y: np.array):
    # Définition de la fenêtre de tolérance pour le contact
    window_range = max(y) - min(y)
    tolerance = window_range * 0.005

    def num_contact_point(slope, intercept, y):
        num = 0
        for i in range(len(y)):
            if abs(y.iloc[i] - (slope * i + intercept)) < tolerance:
                num += 1
        return num

    # Paramètres d'optimisation
    step_slope = 0.01 * init_slope
    step_intercept = 0.01 * init_intercept
    max_iterations = 10
    iteration = 0
    best_slope = init_slope
    best_intercept = init_intercept
    best_contact = num_contact_point(best_slope, best_intercept, y)

    while iteration < max_iterations:
        iteration += 1
        improved = False
        
        # Tester les ajustements autour de la pente et de l'intercept actuels
        for ds in [-step_slope, 0, step_slope]:
            for di in [-step_intercept, 0, step_intercept]:
                test_slope = best_slope + ds
                test_intercept = best_intercept + di
                current_contact = num_contact_point(test_slope, test_intercept, y)
                
                # Mettre à jour si une meilleure configuration est trouvée
                if current_contact > best_contact:
                    best_slope, best_intercept, best_contact = test_slope, test_intercept, current_contact
                    improved = True
        
        # Arrêter si aucune amélioration n'a été trouvée lors de cette itération
        if not improved:
            break

    return best_slope, best_intercept


def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    # coefs[0] = slope,  coefs[1] = intercept
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax() 
    lower_pivot = (low - line_points).argmin() 
    
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    support_coefs = adjust_slope(lower_pivot , support_coefs[0], low)
    resist_coefs = adjust_slope(upper_pivot, resist_coefs[0], high)

    support_coefs = readjust_line(support_coefs[0], support_coefs[1], low)
    resist_coefs = readjust_line(resist_coefs[0], resist_coefs[1], high)

    return (support_coefs, resist_coefs)

def get_srline_points(data, freq=30):
    
    candles = data.iloc[-freq:] # Last freq candles in data
    support_coefs, resist_coefs = fit_trendlines_high_low(candles['high'], candles['low'], candles['close'])
    support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
    resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]
    return support_line, resist_line


def parallel(s_line, r_line, atol=0, rtol=0.03):
    # Calculate the slope and intercept for the line
    s_x = np.arange(len(s_line))
    slope_s, _ = np.polyfit(s_x, s_line, 1)

    r_x = np.arange(len(r_line))
    slope_r, _ = np.polyfit(r_x, r_line, 1)

    # print(slope_s, slope_r)
    # print(abs(slope_s - slope_r))
    # print("Current relative difference:", abs(slope_s - slope_r) / max(abs(slope_s), abs(slope_r)))

    return np.isclose(slope_s, slope_r, atol=atol, rtol=rtol)
    # return abs(slope_s - slope_r) <= tol


def parallel_window(data):
    parallel_window = 0
    for freq in range(10, len(data)):
        s_line, r_line = get_srline_points(data, freq)
        if parallel(s_line, r_line):
            parallel_window=freq
    return parallel_window


def parallel_channel(data):
    window = parallel_window(data)
    if window == 0:
        return None, None

    s_line, r_line = get_srline_points(data, window)
    s_x = np.arange(len(s_line))
    slope_s, _ = np.polyfit(s_x, s_line, 1)
    r_x = np.arange(len(r_line))
    slope_r, _ = np.polyfit(r_x, r_line, 1)

    average_slope = (slope_s + slope_r) / 2
    intercept_s = np.median(s_line) - average_slope * np.median(s_x)
    intercept_r = np.median(r_line) - average_slope * np.median(r_x)

    adjusted_s_line = average_slope * s_x + intercept_s
    adjusted_r_line = average_slope * r_x + intercept_r

    return adjusted_s_line, adjusted_r_line


def parallel_string(data, seq_len):
    to_plot = data[-seq_len:]

    parallel_windows = parallel_channel(to_plot)

    # if any(x is not None for x in parallel_windows):
    if parallel_windows[0] is not None:
        s_line, r_line = parallel_windows
        s_line = np.round(s_line, 2)
        r_line = np.round(r_line, 2)
        s_line = s_line[::5]
        r_line = r_line[::5]
        SR_prompt=f'1. Support Line: This sequence represents the lower boundary of the Bitcoin price range over the considered period. Here is the support line : {s_line}. It is by definition a line.' \
            f'2. Resistance Line: This sequence represents the upper boundary of the Bitcoin price range over the considered period. Here is the support line : {r_line}. It is by definition a line.'
    else:
        SR_prompt = "There are no parallel lines found in this specific period"
    
    return SR_prompt