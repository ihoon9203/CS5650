import pandas as pd
import numpy as np
# initial w = [0,0,0,0]

# standard perceptron algorithm
# df: dataframe to be trained
# w: initial weight = [0,0,0,0]
# epochs: epochs integer
# learning_rate: max 1.0 min 0.0
def standard(df,w,epochs,learning_rate):
    _w = w
    for i in range(epochs):
        df = df.iloc[np.random.permutation(len(df))]
        # iterate through every row in dataframe and update w
        for index, row in df.iterrows():
            X = row[['variance','skewness','curtosis','entropy','bias']].values
            y = row[['label']].values[0]
            _y = np.dot(_w,X)

            if y*_y <= 0: # if y and _y is different sign (added '=' to pass _y=0 case)
                if y > 0 : # positive error case
                    _w = _w+learning_rate*X
                else:
                    _w = _w-learning_rate*X
                
    return _w

# voted perceptron algorithm
# df: dataframe to be trained
# w: initial weight = [0,0,0,0]
# epochs: epochs integer 
# learning_rate: max 1.0 min 0.0
# returns lists of occurance of weights at each index and lists of weights at each index
def voted(df,w,epochs,learning_rate):
    _w = w
    c = [] # life cycle of each weights vectors.
    wlist = [] # stores each weight vectors
    m = 1 # life cycle of current weight vector it's alive at list for 1 life cycle
    for i in range(epochs):
        df = df.iloc[np.random.permutation(len(df))]
        # iterate through every row in dataframe and update w
        for index, row in df.iterrows():
            X = row[['variance','skewness','curtosis','entropy','bias']].values
            y = row[['label']].values[0]
            _y = np.dot(_w,X)

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

# voted perceptron algorithm
# df: dataframe to be trained
# w: initial weight = [0,0,0,0]
# epochs: epochs integer 
# learning_rate: max 1.0 min 0.0
# returns lists of occurance of weights at each index and lists of weights at each index
def average(df,w,epochs,learning_rate):
    _w = w
    a = np.zeros(len(_w)) # average vector have same dimension with weight vector
    c = [] # life cycle of each weights vectors.
    wlist = [] # stores each weight vectors
    m = 1 # life cycle of current weight vector it's alive at list for 1 life cycle
    for i in range(epochs):
        df = df.iloc[np.random.permutation(len(df))]
        # iterate through every row in dataframe and update w
        for index, row in df.iterrows():
            X = row[['variance','skewness','curtosis','entropy','bias']].values
            y = row[['label']].values[0]
            _y = np.dot(_w,X)

            if y*_y <= 0: # if y and _y is different sign (added '=' to pass _y=0 case) --> ERROR!
                if y > 0 : # positive error case
                    _w = _w+learning_rate*X
                else:
                    _w = _w-learning_rate*X
            else: # if no error
                m += 1 # life cycle expands
            a = a + _w
                
    return a

# test for standard perceptron algorithm and average perceptron algorithm
# df: dataframe to be tested
# w: weight for testing
# return correctness max at 1.0
def test(df,w):
    total = df.shape[0]
    error = 0
    for index, row in df.iterrows():
        X = row[['variance','skewness','curtosis','entropy','bias']].values
        y = row[['label']].values[0]
        _y = np.dot(w,X)

        if y*_y < 0: # if y and _y is different sign it is error
            error += 1
    return (total-error)/total

# test for voted perceptron algorithm
# df: dataframe to be tested
# votes: lists of occurance of weights at each index
# voted_w: lists of weights at each index
# return correctness max at 1.0
def test_voted(df,votes,voted_w):
    total = df.shape[0]
    error = 0
    for index, row in df.iterrows():
        X = row[['variance','skewness','curtosis','entropy','bias']].values
        y = row[['label']].values[0]
        sum_vote = 0
        for i in range(len(votes)):
            c = votes[i] # c_i
            w = voted_w[i] # w_i
            sum_vote += c*sign(np.dot(w,X))
        _y = sign(sum_vote)
        if(_y != sign(y)): # if prediction is error, error increment
            error += 1
    return (total-error)/total

# helper method for detecting sign, if input integer is positive, it will return 1, if input integer is either poisitive or zero, it will return -1
# i: integer value to be checked positive or negative
def sign(i):
    if i > 0:
        return 1
    else:
        return -1

