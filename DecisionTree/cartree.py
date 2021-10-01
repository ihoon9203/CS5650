import pandas as pd
import algorithm as al

if __name__ == '__main__':
    df = pd.read_csv("./DecisionTree/Data/car/train.csv",header=None)
    df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boots','safety','labels']
    attributes = {'buying': ['vhigh','high', 'med', 'low'],
                'maint': ['vhigh','high', 'med', 'low'],
                'doors': [2, 3, 4,'5more'],
                'persons': [2, 4, 'more'],
                'lug_boots': ['small', 'med', 'big'],
                'safety': ['low', 'med', 'high']}
    label = ['unacc', 'acc', 'good', 'vgood']
    al.majority_error(df, attributes, label)
    al.gini_index(df,attributes,label)
    al.entropy(df,attributes,label)