# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import mplfinance as mpf

# def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
#     # compute sum of differences between line and prices, 
#     # return negative val if invalid 
    
#     # Find the intercept of the line going through pivot point with given slope

#     intercept = -slope * pivot + y.iloc[pivot]
#     line_vals = slope * np.arange(len(y)) + intercept
     
#     diffs = line_vals - y
    
#     # Check to see if the line is valid, return -1 if it is not valid.
#     if support and diffs.max() > 1e-5:
#         return -1.0
#     elif not support and diffs.min() < -1e-5:
#         return -1.0

#     # Squared sum of diffs between data and line 
#     err = (diffs ** 2.0).sum()
#     return err;


# def optimize_slope(support: bool, pivot:int , init_slope: float, y: np.array):
    
#     # Amount to change slope by. Multiplyed by opt_step
#     slope_unit = (y.max() - y.min()) / len(y) 
    
#     # Optmization variables
#     opt_step = 1.0
#     min_step = 0.0001
#     curr_step = opt_step # current step
    
#     # Initiate at the slope of the line of best fit
#     best_slope = init_slope
#     best_err = check_trend_line(support, pivot, init_slope, y)
#     assert(best_err >= 0.0) # Shouldn't ever fail with initial slope

#     get_derivative = True
#     derivative = None
#     while curr_step > min_step:

#         if get_derivative:
#             # Numerical differentiation, increase slope by very small amount
#             # to see if error increases/decreases. 
#             # Gives us the direction to change slope.
#             slope_change = best_slope + slope_unit * min_step
#             test_err = check_trend_line(support, pivot, slope_change, y)
#             derivative = test_err - best_err;
            
#             # If increasing by a small amount fails, 
#             # try decreasing by a small amount
#             if test_err < 0.0:
#                 slope_change = best_slope - slope_unit * min_step
#                 test_err = check_trend_line(support, pivot, slope_change, y)
#                 derivative = best_err - test_err

#             if test_err < 0.0: # Derivative failed, give up
#                 raise Exception("Derivative failed. Check your data. ")

#             get_derivative = False

#         if derivative > 0.0: # Increasing slope increased error
#             test_slope = best_slope - slope_unit * curr_step
#         else: # Increasing slope decreased error
#             test_slope = best_slope + slope_unit * curr_step
        

#         test_err = check_trend_line(support, pivot, test_slope, y)
#         if test_err < 0 or test_err >= best_err: 
#             # slope failed/didn't reduce error
#             curr_step *= 0.5 # Reduce step size
#         else: # test slope reduced error
#             best_err = test_err 
#             best_slope = test_slope
#             get_derivative = True # Recompute derivative
    
#     # Optimize done, return best slope and intercept
#     return (best_slope, -best_slope * pivot + y.iloc[pivot])


# def fit_trendlines_single(data: np.array):
#     # find line of best fit (least squared) 
#     # coefs[0] = slope,  coefs[1] = intercept 
#     x = np.arange(len(data))
#     coefs = np.polyfit(x, data, 1)

#     # Get points of line.
#     line_points = coefs[0] * x + coefs[1]

#     # Find upper and lower pivot points
#     upper_pivot = (data - line_points).argmax() 
#     lower_pivot = (data - line_points).argmin() 
   
#     # Optimize the slope for both trend lines
#     support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
#     resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

#     return (support_coefs, resist_coefs) 



# def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
#     x = np.arange(len(close))
#     coefs = np.polyfit(x, close, 1)
#     # coefs[0] = slope,  coefs[1] = intercept
#     line_points = coefs[0] * x + coefs[1]
#     upper_pivot = (high - line_points).argmax() 
#     lower_pivot = (low - line_points).argmin() 
    
#     support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
#     resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

#     return (support_coefs, resist_coefs)

# def get_srline_points(data, freq=30):
    
#     candles = data.iloc[-freq:] # Last freq candles in data
#     support_coefs, resist_coefs = fit_trendlines_high_low(candles['high'], candles['low'], candles['close'])
#     support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
#     resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]
#     # result = np.concatenate((support_line, resist_line))
#     return support_line, resist_line

# def draw_lines( data, freq): 
#     candles = data.iloc[-freq:] # Last freq candles in data
#     support_coefs, resist_coefs = fit_trendlines_high_low(candles['high'], candles['low'], candles['close'])
#     support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
#     resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]
#     # Define a custom style for the plot with a white background
#     mc = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='inherit')
#     s = mpf.make_mpf_style(base_mpl_style='ggplot', marketcolors=mc, figcolor='white', gridcolor='gray')

#     # Function to get line points for mplfinance
#     def get_line_points(candles, line_points):
#         idx = candles.index
#         line_i = len(candles) - len(line_points)
#         assert(line_i >= 0)
#         points = []
#         for i in range(line_i, len(candles)):
#             points.append((idx[i], line_points[i - line_i]))
#         return points

#     s_seq = get_line_points(candles, support_line)
#     r_seq = get_line_points(candles, resist_line)

#     print("s_seq: ", s_seq)
#     print("r_seq: ", r_seq)

#     # Plotting with a larger size and white background style
#     mpf.plot(candles, alines=dict(alines=[s_seq, r_seq], colors=['blue', 'red']), type='candle', style=s, figsize=(12, 8))
#     plt.show()

# def vector_lines(data, freq): 
#     candles = data.iloc[-freq:] # Last freq candles in data
#     support_coefs, resist_coefs = fit_trendlines_high_low(candles['high'], candles['low'], candles['close'])
#     support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
#     resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]

#     # Function to get line points for mplfinance
#     def get_line_points(candles, line_points):
#         idx = candles.index
#         line_i = len(candles) - len(line_points)
#         assert(line_i >= 0)
#         points = []
#         for i in range(line_i, len(candles)):
#             points.append((idx[i], line_points[i - line_i]))
#         return points

#     s_seq = get_line_points(candles, support_line)
#     r_seq = get_line_points(candles, resist_line)

#     s_seq = [value for (_, value) in s_seq]
#     r_seq = [value for (_, value) in r_seq]

#     return s_seq, r_seq

# def parallel(s_line, r_line, atol=0, rtol=0.01):
#     # Calculate the slope and intercept for the line
#     s_x = np.arange(len(s_line))
#     slope_s, _ = np.polyfit(s_x, s_line, 1)

#     r_x = np.arange(len(r_line))
#     slope_r, _ = np.polyfit(r_x, r_line, 1)

#     return np.isclose(slope_s, slope_r, atol=atol, rtol=rtol)

# def parallel_channel(data):
#     parallel_windows = 0
#     for freq in range(10, len(data)):
#         s_line, r_line = get_srline_points(data, freq)
#         if parallel(s_line, r_line):
#             parallel_windows=freq
#     return parallel_windows

# def parallel_string(data, freq):
#     to_plot = data[-freq:]

#     parallel_windows = parallel_channel(to_plot)

#     if parallel_windows != 0:
#         s_line, r_line = get_srline_points(to_plot, parallel_windows)
#         s_line = np.round(s_line, 2)
#         r_line = np.round(r_line, 2)
#         s_line = s_line[::5]
#         r_line = r_line[::5]
#         SR_prompt=f'1. Support Line: This sequence represents the lower boundary of the Bitcoin price range over the considered period. Here is the support line : {s_line}. It is by definition a line.' \
#             f'2. Resistance Line: This sequence represents the upper boundary of the Bitcoin price range over the considered period. Here is the support line : {r_line}. It is by definition a line.'
#         return SR_prompt
#     else:
#         return "There are no parallel lines found in this specific period"