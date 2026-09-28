import numpy as np

def compute_loss(y, tx, w):

    """Calculate the loss using either MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D). N denotes the number of samples and D denotes the number of features.
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    N = tx.shape[0]
    return (y - tx@w)@(y - tx@w)/(2*N)


def compute_gradient(y, tx, w):
    """Computes the gradient at w.
    
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D). N denotes the number of samples and D denotes the number of features.
        w: numpy array of shape=(D,). The vector of model parameters.
        
    Returns:
        An numpy array of shape (D,) (same shape as w), containing the gradient of the loss at w.
    """
    N = y.shape[0]
    return -(y-tx@w)@tx/N

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D). N denotes the number of samples and D denotes the number of features.
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: numpy array of shape=(D,). Returns the last weight vector
        loss: A scalar denoting the MSE loss at the final weight vector
    """
    
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx,w)
        
        w = w - gamma*gradient

    #Compute loss at the last weight vector
    loss = compute_loss(y,tx,w)

    return w,loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D). N denotes the number of samples and D denotes the number of features.
        w: numpy array of shape=(D,). The vector of model parameters.
        
    Returns:
        A numpy array of shape (D,) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D). N denotes the number of samples and D denotes the number of features.
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: numpy array of shape=(D,). Returns the last weight vector
        loss: A scalar denoting the MSE loss at the final weight vector
    """
    
    w = initial_w
    
    for _ in range(max_iters):
        # Process one randomly sampled datapoint per iteration.
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=1, num_batches=1
        ):
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad

    # Compute the final loss
    loss = compute_loss(y, tx, w)

    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    N = len(y)
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = (y - tx@w)@(y - tx@w)/(2*N)
    return (w, loss)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    l = lambda_*2*len(y)
    return np.linalg.inv(tx.T@tx + l*np.identity(tx.shape[1]))@tx.T@y

#TODO: Fix ridge_regression (fails 2 tests)
#TODO: Logistic Regression
#TODO : Reg Logistic Regression 
