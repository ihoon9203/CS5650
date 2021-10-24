# ML
This is a machine learning library developed by Jeonghoon Oh for CS5350/6350 in University of Utah

1. algorithm.py
algorithm.py contains algorithm to induce information gain from 3 different ID3 methods which are 
majority error, gini index, and entropy and you can get information gain by importing algorithm.py and
using 'majority_error', 'gini_index', and 'entropy functions'. 
Each function returns dictionary that contains each attribute's information gain.

2. treemaker.py
treemaker.py supposedly uses information gain from the algorithm.py to make a decision tree. It would've 
used recursion for each attributes on dataframe in order of information gain.