###################################################################################
#########################
###################################################################################
import time

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
###################################################################################
#########################  LOAD DATA AND SETUP
###################################################################################
start = time.time()
data = loadmat('../ex3/ex3data1.mat')
X = data['X']
y = data['y']

X, x_test, y, y_test = train_test_split(X, y, test_size=0.3)
y = y.ravel()
y_test = y_test.ravel()
clf = MLPClassifier(hidden_layer_sizes=(25, 25))

clf.fit(X, y) 



y_pred = clf.predict(x_test)

for i in range(len(y_pred)):
    if (i % 500 == 0):
        print y_pred[i]

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y_test)]  
accuracy = (sum(map(int, correct)) / float(len(correct)))  
print 'accuracy = {0}%'.format(accuracy * 100)
end = time.time()
print (end - start)
