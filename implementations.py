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
        loss: 

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N,D = tx.shape
    
    a = tx.T @ tx + 2 * N * lambda_ * np.identity(D)
    b = tx.T @ y

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)

    return w, loss

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    return 1/(1+np.exp(-t))


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    N = tx.shape[0]
    sigmoid_tx_w = sigmoid(tx @ w)
    loss = -np.sum(y * np.log(sigmoid_tx_w) + (1 - y) * np.log(1 - sigmoid_tx_w)) / N
    return np.squeeze(loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    y_col = y.reshape(-1, 1)
    w_col = w.reshape(-1, 1)
    N = tx.shape[0]
    
    pred = sigmoid(tx @ w_col)
    grad = (tx.T @ (pred - y_col)) / N
    
    return grad.reshape(w.shape)

def logistic_regression(y, tx, initial_w, max_iters, gamma) :
    """
        Do one step of Newton's method.
        Return the loss and updated w.
    
        Args:
            y:  shape=(N, 1)
            tx: shape=(N, D)
            w:  shape=(D, 1)
            gamma: scalar
    
        Returns:
            loss: scalar number
            w: shape=(D, 1)
    
        >>> y = np.c_[[0., 0., 1., 1.]]
        >>> np.random.seed(0)
        >>> tx = np.random.rand(4, 3)
        >>> w = np.array([[0.1], [0.5], [0.5]])
        >>> gamma = 0.1
        >>> loss, w = learning_by_newton_method(y, tx, w, gamma)
        >>> round(loss, 8)
        0.71692036
        >>> w
        array([[-1.31876014],
               [ 1.0590277 ],
               [ 0.80091466]])
    """
    w = np.copy(initial_w)
    for _ in range(max_iters):
        grad = calculate_gradient(y, tx, w)
        w = w - gamma * grad

    loss = calculate_loss(y, tx, w)
    return w, np.squeeze(loss)


def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):
    """
        Do one step of gradient descent, using the penalized logistic regression.
        Return the loss and updated w.
    
        Args:
            y:  shape=(N, 1)
            tx: shape=(N, D)
            w:  shape=(D, 1)
            gamma: scalar
            lambda_: scalar
    
        Returns:
            loss: scalar number
            w: shape=(D, 1)
    
        >>> np.set_printoptions(8)
        >>> y = np.c_[[0., 1.]]
        >>> tx = np.arange(6).reshape(2, 3)
        >>> w = np.array([[0.1], [0.2], [0.3]])
        >>> lambda_ = 0.1
        >>> gamma = 0.1
        >>> loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        >>> round(loss, 8)
        0.63537268
        >>> w
        array([[0.10837076],
               [0.17532896],
               [0.24228716]])
    """
    
    w = np.copy(initial_w)
    for _ in range(max_iters):
        gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient

    loss = calculate_loss(y, tx, w)
    return w, np.squeeze(loss)