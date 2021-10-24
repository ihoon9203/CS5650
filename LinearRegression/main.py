import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import gradient_descent as gd

if __name__ == '__main__':
    df = pd.read_csv("./MLisFun/LinearRegression/concrete/train.csv",header=None)
    df.columns = ['Cement','Slag','Fly ash','Water','SP','Coarse Aggr','Fine Aggr','y']
    # batch gradient descent
    lr = [1, 0.5, 0.25, 0.125,0.4,0.3]
    x_bgd = []
    tol = math.pow(10,-6)
    epochs = 40
    learning_rate = 0.5
    ax = []
    ws = []
    ax_s = []
    x_bgds = []
    wss = []
    
    fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2)
    fig.suptitle("BGD Error Graph by change of learning rate")
    length_x = 0
    # plotting cost function value on batch gradient descent and stochastic gradient descent
    for l in lr:
        w, w_error,  err, length = gd.batch(df, epochs, l, tol)
        w_s, w_error, err_s, length_s = gd.stochastic(df, epochs, l, tol)
        length_x = length
        length_xs = length_s
        ax.append(err)
        ax_s.append(err_s)
        x_bgd.append(np.arange(0,length_x))
        x_bgds.append(np.arange(0,length_xs))
        ws.append(w)
        wss.append(w_s)
    fig.suptitle("BGD Error Graph by change of learning rate")
    ax1.plot(x_bgd[0], ax[0], label='1', color='red')
    ax1.title.set_text("learning rate = 1.0")
    ax2.plot(x_bgd[1], ax[1], label='0.5', color='green')
    ax2.title.set_text("learning rate = 0.5")
    ax3.plot(x_bgd[2], ax[2], label='0.25', color='blue')
    ax3.title.set_text("learning rate = 0.25")
    ax4.plot(x_bgd[3], ax[3], label='0.125', color='black')
    ax4.title.set_text("learning rate = 0.125")
    ax5.plot(x_bgd[4], ax[4], label='0.4', color='black')
    ax5.title.set_text("learning rate = 0.4")
    ax6.plot(x_bgd[5], ax[5], label='0.3', color='black')
    ax6.title.set_text("learning rate = 0.3")
    """
    fig.suptitle("SGD Error Graph by change of learning rate")
    ax1.plot(x_bgds[0], ax_s[0], label='1', color='red')
    ax1.title.set_text("learning rate = 1.0")
    ax2.plot(x_bgds[1], ax_s[1], label='0.5', color='green')
    ax2.title.set_text("learning rate = 0.5")
    ax3.plot(x_bgds[2], ax_s[2], label='0.25', color='blue')
    ax3.title.set_text("learning rate = 0.25")
    ax4.plot(x_bgds[3], ax_s[3], label='0.125', color='black')
    ax4.title.set_text("learning rate = 0.125")
    ax5.plot(x_bgds[4], ax_s[4], label='0.4', color='black')
    ax5.title.set_text("learning rate = 0.4")
    ax6.plot(x_bgds[5], ax_s[5], label='0.3', color='black')
    ax6.title.set_text("learning rate = 0.3")
    """
    # weight vector
    # cost function on test data
    df_test = pd.read_csv("./MLisFun/LinearRegression/concrete/test.csv",header=None)
    df_test.columns = ['Cement','Slag','Fly ash','Water','SP','Coarse Aggr','Fine Aggr','y']
    print("learned weight vector for bgd:", ws[3])
    print("cost function for batch gradient descent")
    gd.test_cost(df_test, ws[3])
    print("learned weight vector for sgd:", ws[1])
    print("cost function for stochastic gradient descent")
    gd.test_cost(df_test, wss[1])
    # stochastic gradient descent
    x_sgd = np.arange(0,length_s)
    #plt.plot(x_sgd,err_s, label='SGD error', color='blue')
    plt.show()