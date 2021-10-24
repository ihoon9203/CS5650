import math

# problem 2-2-a
#column label is static
def majority_error(df, attributes, labels):
    nofdata = len(df)
    count = dict()
    # print(type(labels))
    # print(len(attributes))
    # making dict of dict for tracking attribues and values
    # majority error for S
    me_set = df['y'].value_counts()[1]/nofdata
    # print(me_set)
    att_me = dict()
    # for every row, calculate occurence of values in dataframe
    for row in df.iterrows():
        rowlabel = row[1]['y']
        for att in attributes:
            attvalue = row[1][att]
            if count.get((att,attvalue,rowlabel)) is not None:
                count[(att,attvalue,rowlabel)] += 1
            else:
                count[(att,attvalue,rowlabel)] = 1
    
    # for every attributes to 
    for att in attributes:
        valList = attributes.get(att)
        i = 0
        subtractingFactor = 0
        total_val = 0
        for val in valList:
            max = 0
            smax = 0
            max_label = ""
            smax_label = ""
            total_label = 0
            for label in labels:
                # getting majority error case
                if count.get((att,val,label)) is not None:
                    cases = count[(att,val,label)] # number of each attibute value - label
                    total_label+=cases
                    if cases > max:
                        max_label = label
                        smax_label = max_label
                        smax = max
                        max = cases
                    elif cases > smax:
                        smax_label = label
                        smax = cases
                    i += count[(att,val,label)]
                else:
                    count[(att,val,label)] = 0
                total_val += count.get((att,val,label))
            if count.get((att,val,smax_label)) is not None:
                me_p = smax/nofdata
                me_v = count[(att,val,smax_label)]/total_val
                subtractingFactor += me_p*me_v
        att_me[att] = me_set-subtractingFactor # ME for each attribute
    return att_me

def gini_index(df, attributes, labels):
    nofdata = len(df)
    count = dict()
    # print(type(labels))
    # print(len(attributes))
    # making dict of dict for tracking attribues and values
    # majority error for S
    gi_set = 1
    for val in df['y'].value_counts():
        gi_set -= (val/nofdata) ** 2
    # print(me_set)
    att_gi = dict()
    # for every row, calculate occurence of values in dataframe
    for row in df.iterrows():
        rowlabel = row[1]['y']
        for att in attributes:
            attvalue = row[1][att]
            if count.get((att,attvalue,rowlabel)) is not None:
                count[(att,attvalue,rowlabel)] += 1
            else:
                count[(att,attvalue,rowlabel)] = 1
    
    # for every attributes to 
    for att in attributes:
        valList = attributes.get(att)
        i = 0
        subtractingFactor = 0
        total_val = 0
        for val in valList:
            total_label = 0
            gini_p = 0
            gini_v = 0
            total_val = 0
            for label in labels:
                # getting majority error case
                if count.get((att,val,label)) is not None:
                    cases = count[(att,val,label)] # number of each attibute value - label
                    total_label+=cases
                    i += count[(att,val,label)]
                else:
                    count[(att,val,label)] = 0
                total_val += count.get((att,val,label))
            gini_p = total_val/nofdata
            if total_val != 0:
                for label in labels:
                    gini_v += (count[(att,val,label)]/total_val) ** 2
            gini_v = 1 - gini_v
            subtractingFactor += gini_v*gini_p
        att_gi[att] = gi_set-subtractingFactor # GI for each attribute
    return att_gi

def entropy(df, attributes, labels):
    nofdata = len(df)
    base = len(labels)
    count = dict()
    # print(type(labels))
    # print(len(attributes))
    # making dict of dict for tracking attribues and values
    # majority error for S
    entropy_set = 0
    for val in df['y'].value_counts():
        entropy_set += -1*(val/nofdata)*math.log(val/nofdata,base)
    # print(me_set)
    att_entropy = dict()
    # for every row, calculate occurence of values in dataframe
    for row in df.iterrows():
        rowlabel = row[1]['y']
        for att in attributes:
            attvalue = row[1][att]
            if count.get((att,attvalue,rowlabel)) is not None:
                count[(att,attvalue,rowlabel)] += 1
            else:
                count[(att,attvalue,rowlabel)] = 1
    
    # for every attributes to 
    for att in attributes:
        valList = attributes.get(att)
        i = 0
        subtractingFactor = 0
        total_val = 0
        for val in valList: 
            total_label = 0
            entropy_p = 0
            entropy_v = 0
            total_val = 0
            for label in labels:
                # getting majority error case
                if count.get((att,val,label)) is not None:
                    cases = count[(att,val,label)] # number of each attibute value - label
                    total_label+=cases
                    i += count[(att,val,label)]
                else:
                    count[(att,val,label)] = 0
                total_val += count.get((att,val,label))
            entropy_p = total_val/nofdata
            if total_val != 0:
                for label in labels:
                    p = count[(att,val,label)]/total_val
                    if p !=0:
                        entropy_v += -1*p*math.log(p,base)
                subtractingFactor += entropy_p*entropy_v
        att_entropy[att] = entropy_set-subtractingFactor # GI for each attribute
    return att_entropy
