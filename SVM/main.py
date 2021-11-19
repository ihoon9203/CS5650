import pandas as pd
import numpy as np
import os
import gradientdescent as sgd
import test
if __name__ == '__main__':

    dir_path = os.getcwd()
    os.chdir('SVM/bank-note')
    df = pd.read_csv("./train.csv",header=None)
    df.columns = ['variance','skewness','curtosis','entropy','label']
    df['label']=df['label'].replace(0,-1) # replacing 0 to -1 for label values
    # train_X = df[['variance','skewness','curtosis','entropy']]
    # train_y = df[['label']]
    df.insert(df.shape[1]-1,'bias',1)
    test_df = pd.read_csv("./test.csv",header=None)
    test_df.columns = ['variance','skewness','curtosis','entropy','label']
    test_df['label']=test_df['label'].replace(0,-1)
    test_df.insert(df.shape[1]-1,'bias',1)
    # initialization
    w = np.array([0,0,0,0,0])
    max_epochs = 100
    learning_rate = 0.001
    # 100/873, 500/873, 700/873
    hyp_C = {100/873,500/873, 700/873}
    a = 1
    
    # stochastic subgradient descent using learning schedule l = l/(1+(l*t/a))
    """
    for c in hyp_C:
        sub_w = sgd.subgradient_descent(df, w, max_epochs,learning_rate,c, a)
        train_acc = test.test(df,sub_w) # accuracy on training
        test_acc = test.test(test_df,sub_w)
        print('Training accuracy for C value of %d is %d' %(c, train_acc))
        print('Test accuracy for C value of %d is %d' %(c, test_acc))
    """
    # stochastic subgradient descent using learning schedule l = l/(1+t))
    """
    for c in hyp_C:
        sub_w = sgd.subgradient_descent_2(df, w, max_epochs,learning_rate,c, a)
        train_acc = test.test(df,sub_w) # accuracy on training
        test_acc = test.test(test_df,sub_w)
        print(train_acc)
        print(test_acc)
    """