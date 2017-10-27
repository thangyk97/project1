import numpy as np 

def compute_cost(X, y, theta):
    """
    Compute cost of y predict with 'theta' and 'y' 
    Parameters
    ----------
    X : input
    y : label (outcome)
    theta : weight

    Returns
    -------
    Mean square of costs of y predict with 'theta' and y devide to 2.
    Value of cost function

    References
    ----------
    www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
    """
    return np.sum( np.power((X*theta.T-y),2) ) / (2*len(X))

def gradient_descent(X, y, theta, alpha, iters):
    """
    Parameters
    ----------
    X : input
    y : label (outcome)
    theta : init weight 
    alpha : learning rate
    iters : interation number

    Returns
    -------
    theta : weight of minimal cost function
    cost : minimal cost

    References
    ----------
    www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
    """
    temp = theta ## difference between 0 and 0.
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y # (97, 2)

        for j in range(theta.shape[1]):
            term = np.multiply(error, X[:,j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        # end for
        theta = temp
        cost[i] = compute_cost(X, y, theta) # save cost to plot
    #end for
    return theta, cost 

def perform(data, theta, alpha, iters):
    """
    perform set up input and label, training model,
    return output of gradient descent function

    Parameters
    ----------
    data : frame of input and label
    theta : weight initialize
    alpha : learning rate
    iters : Interation number of gradient descent

    Returns
    -------
    theta : weight of minimal cost function
    cost : minimal cost
    
    Author
    ------
    thangyk97@gmail.com
    """
    # Add column of one to data - add bias
    data.insert(0,'bias',1) 

    # get input (X) and label (y)
    # label is right most column
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    # convert from data frames to numpy matrices
    X  = np.matrix(X.values)
    y  = np.matrix(y.values)

    # perform gradient descent to "fit" the model parameters
    g, cost = gradient_descent(X, y, theta, alpha, iters)
    return g, cost