import pandas as pd
import numpy as np
level_df = dict() # key: integer 1-to-depth, value: list of leaf nodes(classified dataframe)

def predictTree(df,attributes,label,max_tree_depth,model):
    sorted_model = [] # order of attributes by model algorithm
    print(model)
    sorted_val = sorted(model, key=model.get, reverse=True)
    print("list:", sorted_model)
    #sorted_model contains attributes to decide
    
    dfs = decision(df,attributes,sorted_val,0,1) # dfs is a stump
    dfs_leaf = dfs[1] # leaf nodes
    for df in dfs:
        adb(df, w)

# gets the stump 
def decision(df, attributes,sorted_val, i, depth):
    
    dataframes = []
    this_att = sorted_val[i] 
    this_labels = attributes[this_att]
    print(this_att)
    for l in this_labels:
        dfs = df[df[this_att]==str(l)] # splitting dataframe by attribute's label
        dataframes.append(dfs)
        if len(dfs['y'].unique()) != 1 and i < depth:
            decision(dfs,attributes,sorted_val,i+1,depth)
    level_df[i] = dataframes
    return level_df

def adb(df, w):
    total_rows = df.shape[0]
    print(total_rows)
    w = np.empty(total_rows)
    w.fill(1/total_rows) # initial weight for adaboosting: equal weight for all
    label = df['y']
    if label == 'no':
