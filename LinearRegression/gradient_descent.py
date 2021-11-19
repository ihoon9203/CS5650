# w0+w0x1+w2x2+w3x3+w4x4+w5x5+w6x6+w7x7=y
import math
import pandas as pd
import numpy as np
import random as rand
def stochastic(df, epochs, r, tol): # dataframe, epochs, learning_rate, tolerance
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
        rand_idx = rand.randint(0, size-1)
        rand_x = X.iloc[[rand_idx]]
        rand_y = y.iloc[[rand_idx]]
        y_exp = np.dot(rand_x,w)+bias
        # gradient for each w_i
        error = rand_y - y_exp
        w_grad = -np.dot(error,rand_x) 
        bias_grad = error.mean()
        # update for w
        new_w = (w - r*w_grad)
        new_bias = (bias -r*bias_grad)
        w_error = new_w-w 
        err = abs(np.linalg.norm(y-np.dot(X,new_w)))
        err_array.append(err)
        if err < tol:
            break
        w = new_w
        bias = new_bias
        
    err_array = np.array(err_array).squeeze()
    return w, w_error, err_array, iterations
    
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
    # use previous attribute vector to calculate new y_hat
    # get the |y_hat - y| -> calculate error based on question instruction? -> error
    # and using that, get gradient descent delta attribute vector
    # using that gradient descent get new attribute vector
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
        err = np.linalg.norm(y-np.dot(X,new_w))
        err_array.append(err)
        if err < tol:
            print(err)
            break
        w = new_w
        bias = new_bias
    
    err_array = np.array(err_array).squeeze()
    return w, w_error, err_array, iterations

def test_cost(df, w):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    y = y.squeeze()
    y_exp = np.dot(X,w)
    err = y - y_exp
    print(np.linalg.norm(err))
    
