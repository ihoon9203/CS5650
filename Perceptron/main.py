import pandas as pd
import numpy as np
import os
import perceptron as perc
if __name__ == '__main__':

    dir_path = os.getcwd()
    df = pd.read_csv("./Perceptron/bank-note/train.csv",header=None)
    df.columns = ['variance','skewness','curtosis','entropy','label']
    df['label']=df['label'].replace(0,-1) # replacing 0 to -1 for label values
    # train_X = df[['variance','skewness','curtosis','entropy']]
    # train_y = df[['label']]

    # initialization
    w = np.array([0,0,0,0])
    epochs = 10
    learning_rate = 0.007
    
    standard_w = perc.standard(df,w,epochs,learning_rate)
    print(standard_w)

    test_df = pd.read_csv("./Perceptron/bank-note/test.csv",header=None)
    test_df.columns = ['variance','skewness','curtosis','entropy','label']
    test_df['label']=test_df['label'].replace(0,-1)
    # test
    performance_standard = perc.test(test_df,standard_w)

    votes, voted_w = perc.voted(df,w,epochs,learning_rate) 
    print(performance_standard)