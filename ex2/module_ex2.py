import numpy as np

def sigmoid(z):
    """
    Parameters
    ----------
    z : number, array, matrix

    Returns
    -------
    Value of sigmoid function """
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    """ cost function
    Parameters
    ----------
    theta : weight
    X : features
    y : label

    Returns
    -------
    Value of cost function.
    """
    theta = np.matrix(theta) #
    X = np.matrix(X)         # Convert to matrix
    y = np.matrix(y)         #

    first = np.multiply(y, np.log(sigmoid(X * theta.T))) # y*log(h_theta(X))
    second = np.multiply((1-y), np.log(1 - sigmoid(X * theta.T))) # (1-y)*log(1 - h_theta(X))
    return -np.sum(first + second) / (len(X)) 

def gradient(theta, X, y):
    """ gradient function
    Parameters
    ----------
    theta : weight
    X : features
    y : label

    Returns
    -------
    grad : the gradient (is not weight)
    """
    theta = np.matrix(theta) #
    X = np.matrix(X)         # Convert to matrix
    y = np.matrix(y)         #

    theta.astype(float)     # To sure theta is float
    grad = theta            # The gradient (is not weight)
    J = sigmoid(X * theta.T) - y # cost

    for i in range(theta.shape[1]): # Loop with all theta_i
        term = np.multiply(J, X[:,i]) # (h_theta(X*theta') - y)*X[:,i]
        grad[i] = np.sum(term) / len(X)
    # End for
    return grad

def predict(theta, X):
    """ predict label from input
    Parameters
    ----------
    theta : weight
    X : features

    Returns
    -------
    predict label
    """
    probability = sigmoid(X * theta.T) # h_theta(X*theta')
    return [ 1 if x >= 0.5 else 0 for x in probability ] # 1 if h_theta(X*theta') >= 0.5
                                                         # else 0

def costReg(theta, X, y, lambda_var):
    """ return cost with regularize
    Parameters
    ----------
    theta : weight
    X : features
    y : label
    lambda_var : regularization parameter

    Returns
    -------
    Value of cost with regularize
    """
    X = np.matrix(X)         #
    y = np.matrix(y)         # Convert to matrix
    theta = np.matrix(theta) #
    
    first = np.multiply(-y, np.log(sigmoid( X * theta.T )) ) # y*log(h_theta(X))
    second = np.multiply(-(1-y), np.log(1 - sigmoid( X*theta.T))) # (1-y)*log(1 - h_theta(X))
    # Regularize 
    reg = lambda_var*(1/(2*len(X)))*np.sum(np.power(theta[1:theta.shape[1]],2))
    return np.sum(first + second) / (2*len(X)) + reg

def gradientReg(theta, X, y, lambda_var):
    """ return the gradient with regularize
    Parameters
    ----------
    theta : weight
    X : features
    y : label
    lambda_var : regularization parameter
    
    Returns
    -------
    The gradient with regularize
    """
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    grad = np.zeros(theta.shape[1])

    error = sigmoid(X * theta.T) - y
    for i in range(theta.shape[1]):
        term = np.multiply(error, X[:,i])
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((lambda_var / len(X)) * theta[:,i])

    return grad