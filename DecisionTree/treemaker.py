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

    for att in attributes.keys():
        print("new attribute")
        print(att)
        for val in attributes[att]:
            print(val)
            adf = df.groupby([att,'labels']).size()[val]
            adf_list = pd.Series(data = adf, index = label)
            max_index = adf_list.idxmax()
            print(max_index)
        count+=1
        if count == max_tree_depth:
            break

def decision(df, attributes,val_list,label, map,sorted_model, i, depth):
    if i == depth:
        return
    att = list(sorted_model.keys())[i]
    listofattributes = list(sorted_model.keys())[:i+1]
    print(att)
    for val in attributes[att]:
        print(val)
        if i + 1 < depth:
            split_df = df.groupby(att)
            new_df = split_df.get_group(str(val))
            indexsList = new_df.index.tolist()
            
            # decision(df, attributes, val_list, label, map, sorted_model, i+1, depth)
        

        
            

    



    # mapping