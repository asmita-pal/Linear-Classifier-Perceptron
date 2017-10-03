# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:35:10 2017

@author: Asmita
"""

import pandas as pd
import matplotlib.pyplot as plt
from numpy import array, dot
import numpy as np
import operator

#Load Data
data = pd.read_csv('perceptrondat1', header = None, delim_whitespace = True)
data2 = pd.read_csv('perceptrondat2', header = None, delim_whitespace = True)

#Plot data
plt.scatter(data[0],data[1])
plt.scatter(data2[0],data2[1])
plt.show()

#Classify data as 0 and 1 and move it to array
new_col = data.shape[1]
data[new_col] ,data2[new_col] = 0, -1 #-1 is a computational requirement
dataset = data.append((-data2), ignore_index = True)
a = dataset.values.tolist()
X,y = [],[]
for row in a:
    X.append([1, row[0],row[1]]) #Vector [1, X]
    y.append(row[2]) #Vector [Y]

#Specify weight vector and learning rate
weights = [-3.0, -1.0, 0.5] #Vector A

#Other sample weight vectors
#weights = [-2.0, -0.9, 0.5]
#weights = [-4.5, -2.5, 0.6]
#weights = random.rand(3)

#Learning rate
eta = 0.6

#Train weights
for i in range(0,50):
    weights_old = weights
    for xi,yi in zip(X, y):
        p, expected = xi, yi
        error = np.zeros(len(xi), float)
        output = dot(weights,p)
        if output < 0:
            error = map(operator.sub, error, xi)
    weights = weights - dot(eta,error)
    if (weights_old[0] == weights[0]):
        count = i
        break
#Plot Classifier        
    fig = plt.figure(figsize =(6,4))
    plt.xlim(-3,4)
    plt.ylim(-3,4)
    plt.scatter(data[0], data[1])
    plt.scatter(data2[0], data2[1])
    m, c = -weights[1]/weights[2] , -weights[0]/weights[2]
    l = np.linspace(-3, 4)
    plt.plot(l, m*l + c , 'k-')
    plt.show()
print "No. of updates for classifying = ", count