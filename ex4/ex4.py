
import numpy as np 
import math
import matplotlib.pyplot as plt 
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score

import module_ex4 as ex4

"""======================= Main ============================"""
data = loadmat('../ex3/ex3data1.mat')
X_data = data['X']
y_data = data['y']

# separate to test and train sets
X_data = np.matrix(X_data)
y_data = np.matrix(y_data)

X, X_test, y, y_test = train_test_split(X_data, y_data,
                                        test_size=0.3,
                                        random_state=42)
# convert y(5000,1) to y(5000,10)
y_one_hot = OneHotEncoder(sparse=False).fit_transform(y)
y_one_hot = np.matrix(y_one_hot)
## Visualize example data
import matplotlib.image as mpimg
img = np.reshape(X[np.random.randint(0, X.shape[0]- 1)],
                 (20,20), 
                 order='F')
plt.figure(figsize= (0.5,0.5))
plt.imshow(img)
# Initialize
input_size = 400
hidden_size = 25
number_labels = 10
learning_rate = 1
epsilon_init = 0.12

init_theta_size = hidden_size * (input_size + 1) + \
                  number_labels * (hidden_size + 1)
init_theta = np.random.random(size=init_theta_size) * \
             2 * epsilon_init - epsilon_init

J, grad = back_propagate(init_theta, input_size, hidden_size, 
                         number_labels, X, y_one_hot, 
                         learning_rate)
fmin = minimize(
    fun = back_propagate, 
    x0 = init_theta,
    args=(input_size, hidden_size, number_labels, 
          X, y_one_hot, learning_rate),
    method='TNC',
    jac=True, 
    options={'maxiter': 250})
a1, z2, a2, z3, h = forward_propagate(fmin.x, input_size, 
                                      hidden_size, number_labels, 
                                      X_test)  
y_pred = np.array(np.argmax(h, axis=1) + 1)  

print accuracy_score(y_test, y_pred)*100

np.savetxt('param.txt', fmin.x, delimiter=',')

plt.show()