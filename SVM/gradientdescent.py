import pandas as pd
import numpy as np

# stochastic subgradient descent using learning schedule l = l/(1+(l*t/a))
def subgradient_descent(df, w, max_epochs, l, c, a):
    # number of training examples
    n = df.shape[0]
    for i in range(0, max_epochs):
        # shuffle dataframe
        df = df.iloc[np.random.permutation(len(df))]
        _l = l
        t = 0
        for index, row in df.iterrows():
            t+=1
            X = row[['variance','skewness','curtosis','entropy','bias']].values
            y = row[['label']].values[0]
            # J(w) = 0.5*np.dot(w,w) + c*max(0,1-y*np.dot(w,X))
            delta_j = w
            if max(0,1-y*np.dot(w,X)) > 0:
                delta_j = w - c*n*y*X
            w = w - _l*delta_j
            _l = _l/(1+(_l*t/a))
    return w

# stochastic subgradient descent using learning schedule l = l/(1+t))
def subgradient_descent_2(df, w, max_epochs, l, c, a):
    # number of training examples
    n = df.shape[0]
    for i in range(0, max_epochs):
        # shuffle dataframe
        df = df.iloc[np.random.permutation(len(df))]
        _l = l
        t = 0
        for index, row in df.iterrows():
            t+=1
            X = row[['variance','skewness','curtosis','entropy','bias']].values
            y = row[['label']].values[0]
            # J(w) = 0.5*np.dot(w,w) + c*max(0,1-y*np.dot(w,X))
            delta_j = w
            if max(0,1-y*np.dot(w,X)) > 0:
                delta_j = w - c*n*y*X
            w = w - _l*delta_j
            # schedule
            _l = _l/(1+t)
    return w
