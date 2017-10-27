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
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1-y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

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
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
 
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
    probability = sigmoid(X * theta.T)
    return [ 1 if x >= 0.5 else 0 for x in probability ]

def costReg(theta, X, y, alpha):
    """ return cost with regularize
    Parameters
    ----------
    theta : weight
    X : features
    y : label
    alpha : learning rate

    Returns
    -------
    Value of cost with regularize
    """
    X = np.matrix(X)
    y = np.matrix(y)
    
    theta = np.matrix(theta)
    
    first = np.multiply( -y, np.log(sigmoid( X * theta.T )) )
    second = np.multiply( -(1-y), np.log( 1 - sigmoid( X*theta.T)) )
    reg = alpha*(1/(2*len(X)))*np.sum(np.power(theta[0:theta.shape[1]],2))
    return np.sum(first + second) / (2*len(X)) + reg

def gradientReg(theta, X, y, alpha):
    """ return the gradient with regularize
    Parameters
    ----------
    theta : weight
    X : features
    y : label
    alpha : learning rate
    
    Returns
    -------
    The gradient with regularize
    """
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1]) # unroll
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((alpha / len(X)) * theta[:,i])

    return grad