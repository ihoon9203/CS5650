import pandas as pd

def predictTree(df,attributes,label,max_tree_depth,model):
    sorted_model = [] # order of attributes by model algorithm
    print(model)
    sorted_val = sorted(model, key=model.get, reverse=True)
    for val in sorted_val:
        sorted_model.append(val)
    print("list:", sorted_model)
    #sorted_model contains attributes to decide
    
    # training
    count = 0
    map = dict()
    val_list = []
    val_labels = dict()
    decision(df,attributes,val_list,label,map,sorted_model,0,max_tree_depth)


def decision(df, attributes,sorted_model, i, depth):
    
    dataframes = []
    this_att = sorted_model[i] 
    this_labels = attributes[this_att]
    print(this_labels)
    for l in this_labels:
        dfs = df[df[this_att]==str(l)] # splitting dataframe by attribute's label
        dataframes.append(dfs)
        if len(dataframes) == 1 or i == depth:
            return
    for dataframe in dataframes: # for every leaf in the tree do another decision tree
        decision(dataframe, attributes,sorted_model, i, depth)
        

        
            

    



    # mapping