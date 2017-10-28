import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.optimize as opt 
import os 

import module_ex2 as ex2

data = pd.read_csv('ex2data1.txt', header=None,
                   names=['Exam_1', 'Exam_2', 'Admitted'])
# Show data header
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

# Add a ones column
data.insert(0,'Bias', 1)

# Set X (training data) and y ( target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# Convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

result = opt.fmin_tnc(func=ex2.cost, x0=theta, fprime=ex2.gradient, args=(X,y) )

theta_min = np.matrix(result[0])
predictions = ex2.predict(theta_min, X)

# Print predictions
correct = [1 if ((a == 1 and b == 1) or (a==0 and b == 0)) else 0
           for (a,b) in zip(predictions, y)]
accurancy = (sum(map(int, correct)) % len(correct))
print 'accurancy = {0}%'.format(accurancy)

# plot data
y = np.zeros((1,2))
x = np.array([np.min(X[:,1]), np.max(X[:,1])])

y[0, 0] = - (theta_min[0,0] + theta_min[0,1]*x[0]) / theta_min[0,2]
y[0, 1] = - (theta_min[0,0] + theta_min[0,1]*x[1]) / theta_min[0,2]

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(positive['Exam_1'], positive['Exam_2'],
           s= 50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam_1'], negative['Exam_2'],
           s= 50, c='r', marker='x', label='Not Admitted')
plt.plot(x, np.ravel(y), c='k', label='Decision boundary')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')

plt.show()