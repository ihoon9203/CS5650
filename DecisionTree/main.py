import pandas as pd
import algorithm as al
import treemaker as tree
import adaboost as ad
import os
if __name__ == '__main__':
    #df = pd.read_csv("./MLisFun/DecisionTree/Data/car/train.csv",header=None)
    #df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boots','safety','labels']
    #attributes = {'buying': ['vhigh','high', 'med', 'low'],
    #            'maint': ['vhigh','high', 'med', 'low'],
    #            'doors': [2, 3, 4,'5more'],
    #            'persons': [2, 4, 'more'],
    #            'lug_boots': ['small', 'med', 'big'],
    #            'safety': ['low', 'med', 'high']}
    #label = ['unacc', 'acc', 'good', 'vgood']
    #me = al.majority_error(df, attributes, label)
    #gini = al.gini_index(df,attributes,label)
    #entropy = al.entropy(df,attributes,label)
    #tree.predictTree(df,attributes,label,1,entropy)
    df_bank= pd.read_csv("./MLisFun/DecisionTree/Data/bank/train.csv",header=None)
    df_bank.columns = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
    
    # numerical-data > boolean processing
    age_med = df_bank['age'].median()
    balance_med = df_bank['balance'].median()
    day_med = df_bank['day'].median()
    duration_med =df_bank['duration'].median()
    previous_med = df_bank['previous'].median()
    df_pdays = df_bank[df_bank['pdays']!=-1] # splitting dataframe by attribute's label
    pdays_med = df_pdays['pdays'].median()
    campaign_med = df_bank['campaign'].median()
    for i in df_bank.index:
        if df_bank.at[i,'age'] < age_med:
            df_bank.at[i,'age'] = 0
        else:
            df_bank.at[i,'age'] = 1
        if df_bank.at[i,'balance'] < balance_med:
            df_bank.at[i,'balance'] = 0
        else: 
            df_bank.at[i,'balance'] = 1
        if df_bank.at[i,'day'] < day_med:
            df_bank.at[i,'day'] = 0
        else:
            df_bank.at[i,'day'] = 1
        if df_bank.at[i,'duration'] < duration_med:
            df_bank.at[i,'duration'] = 0
        else:
            df_bank.at[i,'duration'] = 1
        if df_bank.at[i,'previous'] < previous_med:
            df_bank.at[i,'previous'] = 0
        else:
            df_bank.at[i,'previous'] = 1
        if df_bank.at[i,'pdays'] < pdays_med: # -1: not contacted, 0: less then mean, 1: greater then mean
            df_bank.at[i,'pdays'] = 0
        else:
            df_bank.at[i,'pdays'] = 1
        if df_bank.at[i,'campaign'] < campaign_med:
            df_bank.at[i,'campaign'] = 0
        else:
            df_bank.at[i,'campaign'] = 1
    attributes = {'age': [0,1],
                'job': ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services"],
                'marital': [ "married","divorced","single"],
                'education': ["unknown","secondary","primary","tertiary"],
                'default': ['yes','no'],
                'balance': [0,1],
                'housing': ['yes','no'],
                'loan': ['yes','no'],
                'contact':["unknown","telephone","cellular"],
                'day':[0,1],
                'month':['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
                'duration':[0,1],
                'campaign':[0,1],
                'pdays':[-1,0,1],
                'previous':[0,1],
                'poutcome':["unknown","other","failure","success"]}
    label = ['yes','no']
    me = al.majority_error(df_bank, attributes, label)
    gini = al.gini_index(df_bank,attributes,label)
    entropy = al.entropy(df_bank,attributes,label)
    df_nodes = ad.predictTree(df_bank,attributes,label,0,me)
    l = 1