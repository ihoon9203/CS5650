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

3. gradient_descent.py
gradient_Descent.py supposedly contains function for stochastic gradient descent and batch gradient descent,
which could be called each by 'batch(dataframe, epochs, learning_rate, tolerance)' and '
stochastic(dataframe, epochs, learning_rate, tolerance)'.
However, in high probability there are faultiness with these methods that would make the wrong result. 

4. Perceptron
perceptron.py: this file contains training methods and testing methods for 3 perceptron algorithm 
for the 3 training methods, each are standard, voted, and average. These methods trains the given dataset
by using perceptron algorithm of their corresponding name. 
They all take same parameters which are dataframe, initial weight vector,epochs and learning rate
which can be set in the main.py file and further details are commented inside the methods.
There exists two test methods each are named as test and test_voted. 
test method takes dataframe and weight vector as the parameter and tests the dataframe using the weight vector created by
standard perceptron algorithm and average perceptron algorithm.
test_voted takes dataframe and list of weight vectors and list of their occurence as the parameter. This methods tests the
dataframe using voted perceptron algorithm.
in the main.py you can set learning rate and number of epochs. If you run the main.py file, you can
get the weight vectors of each perceptron algorithm learned from training dataset and can test their accuracy on the
test dataset. 
