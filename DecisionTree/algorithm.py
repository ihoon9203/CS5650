import pandas as pd

def majority_error(df, attributes, labels):
    nofdata = len(df)
    
    # majority error for S
    me_set = df['labels'].value_counts()[1]

    #for every row, calculate occurence of values in dataframe
    for att in attributes:
        print(df.groupby(att).count())
    for val in df['labels'].value_counts():
        print(val)

    print(type(df['labels'].value_counts()))