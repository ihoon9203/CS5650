import pandas as pd
import numpy as np
import os
import csv
import perceptron as perc

import time
if __name__ == '__main__':

    dir_path = os.getcwd()
    print(dir_path)
    df = pd.read_csv("./Assignment/Perceptron/bank-note/train.csv",header=None)
    df.columns = ['variance','skewness','curtosis','entropy','label']
    df['label']=df['label'].replace(0,-1) # replacing 0 to -1 for label values
    # train_X = df[['variance','skewness','curtosis','entropy']]
    # train_y = df[['label']]
    df.insert(df.shape[1]-1,'bias',1)
    # initialization
    w = np.array([0,0,0,0,0])
    epochs = 10
    learning_rate = 0.123
    
    # standard perceptron weight vector training
    standard_w = perc.standard(df,w,epochs,learning_rate)
    standard_start = time.time()
    print("Standard Perceptron Weight Vectors: ", standard_w)
    standard_end = time.time()
    print("time taken: ", standard_end-standard_start)
    # voted perceptron weight vectors training
    voted_start = time.time()
    votes, voted_w = perc.voted(df,w,epochs,learning_rate)
    voted_end = time.time()
    print("Number of weight vectors: ",len(votes))
    print("time taken: ", voted_end-voted_start)
    with open('voted_weights.csv', 'w',newline="") as csvfile:
        vote_writer = csv.writer(csvfile)
        vote_writer.writerows(voted_w)

    # average perceptron weight vectors training
    avg_start = time.time()
    average_w = perc.average(df,w,epochs,learning_rate)
    avg_end = time.time()
    print("Average Perceptron Weight Vectors:", average_w)
    
    print("time taken: ", avg_end-avg_start)
    test_df = pd.read_csv("./Assignment/Perceptron/bank-note/test.csv",header=None)
    test_df.columns = ['variance','skewness','curtosis','entropy','label']
    test_df['label']=test_df['label'].replace(0,-1)
    test_df.insert(df.shape[1]-1,'bias',1)
    print(test_df.shape)
    # test
    # making prediction for standard weights
    performance_standard = perc.test(test_df,standard_w)
    # making prediction for voted weights
    performance_voted = perc.test_voted(test_df,votes,voted_w)
    # making prediction for average weights
    performance_average = perc.test(test_df,average_w)
    
    print("accuracy of standard perceptron algorithm: ", performance_standard)
    print("accuracy of voted perceptron algorithm: ", performance_voted)
    print("accuracy of average perceptron algorithm: ", performance_average)