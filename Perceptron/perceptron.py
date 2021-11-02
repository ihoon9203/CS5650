import pandas as pd
import numpy as np
import math
# initial w = [0,0,0,0]
def standard(df,w,epochs,learning_rate):
    _w = w
    for i in range(epochs):
        # iterate through every row in dataframe and update w
        for index, row in df.iterrows():
            X = row[['variance','skewness','curtosis','entropy']].values
            y = row[['label']].values[0]
            _y = np.dot(w,X)

            if y*_y <= 0: # if y and _y is different sign (added '=' to pass _y=0 case)
                if y > 0 : # positive error case
                    _w = _w+learning_rate*X
                else:
                    _w = _w-learning_rate*X
                
    return _w

def voted(df,w,epochs,learning_rate):
    _w = w
    c = [] # life cycle of each weights vectors.
    wlist = []
    m = 1 # life cycle of current weight vector it's alive at list for 1 life cycle
    for i in range(epochs):
        # iterate through every row in dataframe and update w
        for index, row in df.iterrows():
            X = row[['variance','skewness','curtosis','entropy']].values
            y = row[['label']].values[0]
            _y = np.dot(w,X)

            if y*_y <= 0: # if y and _y is different sign (added '=' to pass _y=0 case) --> ERROR!
                wlist.append(_w) # saves weight vector that finished it's life cycle to index
                c.append(m) # weight vector at index is going to be updated and it's life cycle ends
                m = 1 # reset life cycle to 1
                if y > 0 : # positive error case
                    _w = _w+learning_rate*X
                else:
                    _w = _w-learning_rate*X
            else: # if no error
                m += 1 # life cycle expands
                
    return c, wlist

def test(df,w):
    total = df.shape[0]
    error = 0
    for index, row in df.iterrows():
            X = row[['variance','skewness','curtosis','entropy']].values
            y = row[['label']].values[0]
            _y = np.dot(w,X)

            if y*_y < 0: # if y and _y is different sign it is error
                error += 1
    return (total-error)/total
    