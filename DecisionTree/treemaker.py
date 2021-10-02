import pandas as pd

def predictTree(df,attributes,label,max_tree_depth,model):
    sorted_model = {}
    sorted_val = sorted(model, key=model.get, reverse=True)
    for val in sorted_val:
        sorted_model[val] = model[val]
    print(sorted_model)
    #sorted_model contains attributes to decide
    
    # training
    count = 0
    map = dict()
    val_list = []
    val_labels = dict()
    decision(df,attributes,val_list,label,map,sorted_model,0,max_tree_depth)

def decision(df, attributes,val_list,label, map,sorted_model, i, depth):
    if i == depth:
        return
    att = list(sorted_model.keys())[i]
    listofattributes = list(sorted_model.keys())[:i+1]
    print(att)
    df_dict = dict()
    for val in attributes[att]:
        print(val)
        split_df = df.groupby(att)
        new_df = split_df.get_group(str(val))
        length = len(new_df.index)
        
            # decision(df, attributes, val_list, label, map, sorted_model, i+1, depth)
    if i + 1 < depth:
        indexList = new_df.index.tolist()
        decision(new_df, attributes,map,sorted_model, i+1, depth)
        
        

        
            

    



    # mapping