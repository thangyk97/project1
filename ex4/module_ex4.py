import numpy as np 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagate(
    init_theta,input_size,
    hidden_size,number_labels,
    X):
    theta1 = np.array(np.reshape(
        init_theta[:hidden_size*(input_size + 1)],
        (hidden_size, (input_size + 1))
    ))
    theta2 = np.array(np.reshape(
        init_theta[hidden_size*(input_size + 1):],
        (number_labels, (hidden_size + 1))
    ))

    ones = np.ones((X.shape[0],1))
    a1 = np.concatenate((ones, X), axis=1)
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((z2.shape[0], 1)), a2), axis=1)
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

def cost(
    init_theta, input_size, hidden_size,
    number_labels, X, y, learning_rate):

    theta1 = np.array(np.reshape(
        init_theta[:hidden_size*(input_size + 1)],
        (hidden_size, (input_size + 1))
    ))
    theta2 = np.array(np.reshape(
        init_theta[hidden_size*(input_size + 1):],
        (number_labels, (hidden_size + 1))
    ))

    X = np.array(X)
    a1, z2, a2, z3, h = \
        forward_propagate(init_theta, input_size,
                          hidden_size, number_labels, X)
    J = np.sum(
            np.multiply(-y, np.log(h)) -
            np.multiply(1-y, np.log(1 - h)))
    reg = (np.sum(theta1**2) + np.sum(theta2**2)) * learning_rate / 2
    J += reg
    J = J / X.shape[0]
    return J

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))

def back_propagate(
    init_theta, input_size, hidden_size,
    number_labels, X, y, learning_rate):
    theta1 = np.array(np.reshape(
        init_theta[:hidden_size*(input_size + 1)],
        (hidden_size, (input_size + 1))
    ))
    theta2 = np.array(np.reshape(
        init_theta[hidden_size*(input_size + 1):],
        (number_labels, (hidden_size + 1))
    ))

    a1, z2, a2, z3, h = \
        forward_propagate(init_theta, input_size,
                       hidden_size, number_labels, X)
    # compute cost
    J = np.sum(
        np.multiply(-y, np.log(h)) - 
        np.multiply(1-y, np.log(1 - h)))
    reg = (np.sum(theta1**2) + np.sum(theta2**2)) * learning_rate / 2
    J += reg
    J = J / X.shape[0]

    Delta1 = np.zeros((z2.shape[1], a1.shape[1]))
    Delta2 = np.zeros((h.shape[1], a2.shape[1]))
    
    for i in range(X.shape[0]):
        a1t = a1[i,:].T # (401, 1) 
        z2t = z2[i,:].T # (25, 1)
        a2t = a2[i,:].T # (26, 1)
        z3t = z3[i,:].T # (10, 1)
        ht = h[i,:].T  # (10, 1)
        yt = y[i,:].T   # (10, 1)

        d3t = ht - yt # (10, 1)
        d2t = np.multiply( np.dot(theta2[:,1:].T, d3t), 
                           sigmoid_gradient(z2t)) # (25,1)
        
        Delta1 = Delta1 + np.dot(d2t, a1t.T)
        Delta2 = Delta2 + np.dot(d3t, a2t.T)

    Delta1 = Delta1 / X.shape[0]
    Delta2 = Delta2 / X.shape[0]
    grad = np.concatenate((np.ravel(Delta1), np.ravel(Delta2)))
    return J, grad