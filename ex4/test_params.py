import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import loadmat

data = loadmat('../ex3/ex3data1.mat')
X_data = data['X']
y_data = data['y']

import matplotlib.image as mpimg
sample = X_data[np.random.randint(0, X_data.shape[0]- 1)]
img = np.reshape(sample,(20,20), order='F')
plt.figure(figsize= (0.5,0.5))
plt.imshow(img)
 
param = np.loadtxt('param.txt')

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_prop(param, input_size, hidden_size, number_label, X):
    theta1 = np.reshape(param[: hidden_size*(input_size + 1)], (hidden_size, input_size + 1))
    theta2 = np.reshape(param[hidden_size*(input_size + 1) :], (number_label, hidden_size + 1))
    
    X = np.concatenate((np.array([1]),X),axis=0)
    z2 = np.dot(X,theta1.T)
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.array([1]),a2), axis=0)
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)

    return np.array(np.argmax(h, axis=0) + 1)

print (forward_prop(param, 400, 25, 10, sample))

plt.show()