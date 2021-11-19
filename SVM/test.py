import pandas as pd
import numpy as np
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
