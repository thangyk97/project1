#import library
import os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd

import module_ex2 as ex2

#read data
data = pd.read_csv('ex2data2.txt', header=None, 
                   names=['Test 1','Test 2','Accepted'])

#display sample data
print data.head()
positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]

fig, ax = plt.subplots()
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, 
           c='b', marker='o', label='positive')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, 
           c='r', marker='x', label='negative')
plt.legend()
ax.set_xlabel('Test 1')
ax.set_ylabel('Test 2')
ax.set_title('test chip')

#feed into the classifier
degree = 5
x1 = data['Test 1']
x2 = data['Test 2']

data.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data.drop('Test 1', axis=1, inplace=True)
data.drop('Test 2', axis=1, inplace=True)

print data.head() 

# set up features, label and weight
cols = data.shape[1]
X = data.iloc[:,1: cols]
y = data.iloc[:,0:1]

X = np.array(X.values)
y = np.array(y.values)

theta = np.zeros(11)

learningRate = 1

# Optimize 
result = opt.fmin_tnc(func=ex2.costReg, x0=theta, 
                      fprime=ex2.gradientReg, 
                      args=(X, y, learningRate))

# Predict
theta_min = np.matrix(result[0])
predictions = ex2.predict(theta_min, X)
correct = [ 1 if ((a==1 and b==1) or (a==0 and b==0)) else 0 
            for (a,b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))

#display result
print 'accuracy = {0}%'.format(accuracy)
plt.show()