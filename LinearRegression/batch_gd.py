# w0+w0x1+w2x2+w3x3+w4x4+w5x5+w6x6+w7x7=y
import math
import pandas as pd
import numpy as np
def batch(df, epochs, r, tol): # dataframe, epochs, learning_rate, tolerance
    w = np.zeros(7) # weight w1, ..., w7
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    y = y.squeeze()
    bias = 0
    err_array = []
    error = 0
    iterations = 0
    size = df.shape[0]
    for i in range(0,epochs):
        iterations += 1
        y_exp = np.dot(X,w)+bias

        # gradient for each w_i
        error = y - y_exp
        w_grad = -np.dot(error,X) 
        bias_grad = error.mean()
        # update for w
        new_w = (w - r*w_grad)/size
        new_bias = (bias -r*bias_grad)/size
        w_error = new_w-w 
        err = np.linalg.norm(w_error)
        err_array.append(err)
        if err < tol:
            break
        w = new_w
        bias = new_bias
        
    err_array = np.array(err_array).squeeze()
    return w, w_error, err_array, iterations
        