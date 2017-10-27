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

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 50), random_state=1, activation="logistic")

clf.fit(X,y)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, y_pred)*100