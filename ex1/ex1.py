# Linear regression

import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

import module_ex1 as ex1


""" =================== Main ex1data1 ============================ """
# import data and visualize data
data = pd.read_csv('ex1data1.txt', header=None,names=['Population','Profit'])
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))

alpha = 0.01 # Learning rate
iters = 1000 # Interation number of gradient descent
g, cost = ex1.perform(data, np.zeros((1, 2)), alpha, iters)

# view the results
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0,0] + (g[0,1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training data')
ax.legend(loc = 2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted profit vs. Population Size')

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

"""        ==================Main ex1data2====================          """
data = pd.read_csv('ex1data2.txt', header = None, names=['Size','Bedrooms','Price'])
data = (data - data.mean()) / data.std() # features normalization

#define
alpha = 0.01
iters = 750
g, cost = ex1.perform(data, np.zeros((1, 3)), alpha, iters)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch aaa')

plt.show()
