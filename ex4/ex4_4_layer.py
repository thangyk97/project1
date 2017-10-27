
import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt 
from scipy.io import loadmat

## Data
data = loadmat('../ex3/ex3data1.mat')
X_data = data['X']
y_data = data['y']

# separate to test and train sets
from sklearn.model_selection import train_test_split
X, X_test, y, y_test = train_test_split( X_data, y_data, test_size=0.3, random_state=42)

# convert y(5000,1) to y(5000,10)
from sklearn.preprocessing import OneHotEncoder
y_one_hot = OneHotEncoder(sparse=False).fit_transform(y)

## Visualize example data
import matplotlib.image as mpimg
img = np.reshape(X[np.random.randint(0, X.shape[0]- 1)],(20,20), order='F')
plt.figure(figsize= (0.5,0.5))
plt.imshow(img)

## Function
def sigmoid(z):
    
    return 1 / (1 + np.exp(-z))

def forward_propagate(init_theta, input_size, hidden_size, number_labels, X):
    theta1 = np.array(np.reshape(
        init_theta[:hidden_size*(input_size + 1)],
        (hidden_size, (input_size + 1))
    ))
    theta2 = np.array(np.reshape(
        init_theta[hidden_size*(input_size + 1):hidden_size*(input_size + 1) + hidden_size*(hidden_size + 1)],
        (hidden_size, (hidden_size + 1))
    ))
    theta3 = np.array(np.reshape(
        init_theta[hidden_size*(input_size + 1) + hidden_size*(hidden_size + 1) :],
        (number_labels, hidden_size + 1)
    ))
    
    X = np.matrix(X)

    ones = np.ones((X.shape[0],1))
    a1 = np.concatenate((ones, X), axis=1)
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((z2.shape[0], 1)), a2), axis=1)
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    a3 = np.concatenate((np.ones((z3.shape[0], 1)), a3), axis=1)
    z4 = np.dot(a3, theta3.T)
    h = sigmoid(z4)

    return a1, z2, a2, z3, a3, z4, h

def cost(init_theta, input_size, hidden_size, number_labels, X, y, learning_rate):

    theta1 = np.array(np.reshape(
        init_theta[:hidden_size*(input_size + 1)],
        (hidden_size, (input_size + 1))
    ))
    theta2 = np.array(np.reshape(
        init_theta[hidden_size*(input_size + 1):hidden_size*(input_size + 1) + hidden_size*(hidden_size + 1)],
        (hidden_size, (hidden_size + 1))
    ))
    theta3 = np.array(np.reshape(
        init_theta[hidden_size*(input_size + 1) + hidden_size*(hidden_size + 1) :],
        (number_labels, hidden_size + 1)
    ))

    X = np.array(X)
    a1, z2, a2, z3, a3, z4, h = forward_propagate(init_theta, input_size, hidden_size, number_labels, X)
    J = np.sum(np.multiply(-y, np.log(h)) - np.multiply(1-y, np.log(1 - h)))
    reg = (np.sum(theta1**2) + np.sum(theta2**2) + np.sum(theta3**2)) * learning_rate / 2
    J += reg
    J = J / X.shape[0]

    return J

def sigmoid_gradient(z):

    return np.multiply(sigmoid(z), 1 - sigmoid(z))

def back_propagate(init_theta, input_size, hidden_size, number_labels, X, y, learning_rate):
    theta1 = np.array(np.reshape(
        init_theta[:hidden_size*(input_size + 1)],
        (hidden_size, (input_size + 1))
    ))
    theta2 = np.array(np.reshape(
        init_theta[hidden_size*(input_size + 1):hidden_size*(input_size + 1) + hidden_size*(hidden_size + 1)],
        (hidden_size, (hidden_size + 1))
    ))
    theta3 = np.array(np.reshape(
        init_theta[hidden_size*(input_size + 1) + hidden_size*(hidden_size + 1) :],
        (number_labels, hidden_size + 1)
    ))

    X = np.matrix(X)
    y = np.matrix(y)

    a1, z2, a2, z3, a3, z4, h = \
     forward_propagate(init_theta, input_size, hidden_size, number_labels, X)
    print h.shape
    # compute cost
    J = np.sum(np.multiply(-y, np.log(h)) - np.multiply(1-y, np.log(1 - h)))
    reg = (np.sum(theta1**2) + np.sum(theta2**2) + np.sum(theta3**2)) * learning_rate / 2
    J += reg
    J = J / X.shape[0]

    Delta1 = np.zeros((z2.shape[1], a1.shape[1]))
    Delta2 = np.zeros((z3.shape[1], a2.shape[1]))
    Delta3 = np.zeros((h.shape[1], a3.shape[1]))
    
    for i in range(X.shape[0]):
        a1t = a1[i,:].T # (401, 1) 
        z2t = z2[i,:].T # (25, 1)
        a2t = a2[i,:].T # (26, 1)
        z3t = z3[i,:].T # (25, 1)
        a3t = a3[i,:].T # (26, 1)
        z4t = z4[i,:].T # (10, 1)
        ht = h[i,:].T  # (10, 1)
        yt = y[i,:].T   # (10, 1)

        d4t = ht - yt # (10, 1)
        d3t = np.multiply(np.dot(theta3[:,1:].T, d4t), sigmoid_gradient(z3t)) # (25,1)
        d2t = np.multiply(np.dot(theta2[:,1:].T, d3t), sigmoid_gradient(z2t)) # (25,1)
        Delta1 = Delta1 + np.dot(d2t, a1t.T)
        Delta2 = Delta2 + np.dot(d3t, a2t.T)
        Delta3 = Delta3 + np.dot(d4t, a3t.T)

    Delta1 = Delta1 / X.shape[0]
    Delta2 = Delta2 / X.shape[0]
    Delta3 = Delta3 / X.shape[0]
    grad = np.concatenate((np.ravel(Delta1), np.ravel(Delta2), np.ravel(Delta3)))

    return J, grad

## Main
input_size = 400
hidden_size = 25
number_labels = 10
learning_rate = 1
epsilon_init = 0.12

init_theta_size = hidden_size * (input_size + 1) + hidden_size * (hidden_size + 1) + number_labels * (hidden_size + 1)
init_theta = np.random.random(size=init_theta_size) * 2 * epsilon_init - epsilon_init

J, grad = back_propagate(init_theta, input_size, hidden_size, number_labels, X, y_one_hot, learning_rate)

# Optimize
from scipy.optimize import minimize
fmin = minimize(fun = back_propagate, x0 = init_theta,
        args=(input_size, hidden_size, number_labels, X, y_one_hot, learning_rate),
        method='TNC', jac=True, options={'maxiter': 250})
print fmin.x.shape

a1, z2, a2, z3, a3, z4, h = forward_propagate(fmin.x, input_size, hidden_size, number_labels, X_test)  
y_pred = np.array(np.argmax(h, axis=1) + 1)  
print y_pred

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, y_pred)*100

np.savetxt('param.txt', fmin.x, delimiter=',')
