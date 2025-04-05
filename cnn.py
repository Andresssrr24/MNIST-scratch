import numpy as np

def zero_pad(X, pad):
    '''
    Apply padding to a multichannel volume

    Inputs:
    X: numpy.ndarray of shape (m, nH, nW, nC)
    pad: int, padding size

    Returns:
    X padded
    '''
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0,0))

    return X_pad

def forward(A_prev, W, b, hparameters):
    '''
    Forward propagation of a ConvNet (Convolve input volume by n-dimensional filters + biases)

    Inputs:
    A_prev: np.ndarray -> Activation output from the previous layer of shape (m, nH, nW, nC)
    W: np.ndarray -> Filters of shape (f, f, nC_prev, nC) 
    b: np.ndarray -> Biases from the current layer of shape (1, 1, 1, nC)
    hparameters: Dictionary with stride and padding values

    Returns:
    Z: np.ndarray of shape (m, nH, nW, nC)
    cache: tuple -> Saves (A_prev, W, b, hparameters) for backpropagation
    '''
    (m, nH_prev, nW_prev, nC_prev) = A_prev.shape
    (f, f, nC_prev, nC) = W.shape

    stride, pad = hparameters['stride'], hparameters['pad']

    nH = int(((nH_prev - f + 2 * pad)/stride) + 1)
    nW = int(((nW_prev - f + 2 * pad)/stride) + 1)

    Z = np.zeros((m, nH, nW, nC))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]

        for h in range(nH):
            vertical_start = h * stride
            vertical_end = vertical_start + f

            for w in range(nW):
                horizontal_start = w * stride
                horizontal_end = horizontal_start + f

                for c in range(nC):
                    a_slice_prev = a_prev_pad[vertical_start:vertical_end, horizontal_start:horizontal_end, :]

                    weights = W[:, :, : , c]
                    biases = b[:, :, : , c]

                    Z[i, h, w, c] = np.sum(a_slice_prev *  weights) + biases

    cache = (A_prev, W, b, hparameters)

    return Z, cache

def max_pooling(A_prev, hparameters):
    '''
    Apply forward Max Pooling 

    Inputs:
    A_prev: np.ndarray -> Input volume of shape (m, nH_prev, nW_prev, nC_prev)
    hparameters: dictionary containing padding and stride values

    Returns:
    A = np.ndarray -> Output of the pool layer of shape (m, nH, nW, nC)
    cache = tuple for backpropagation in pooling layer
    '''
    (m , nH_prev, nW_prev, nC_prev) = A_prev.shape

    f = hparameters['f']
    stride = hparameters['stride']

    # Set output's dimension
    nH = int(1 + (nH_prev - f) / stride)
    nW = int(1 + (nH_prev - f) / stride)
    nC = nC_prev

    A = np.zeros((m, nH, nW, nC))

    for i in range(m):
        for h in range(nH):
            vertical_start = h * stride
            vertical_end = vertical_start + 1

            for w in range(nW):
                horizontal_start = w * stride
                horizontal_end = horizontal_start + 1

                for c in range(nC):
                    a_prev_slice = A_prev[i, vertical_start:vertical_end, horizontal_start:horizontal_end, c]

                    A[i, h, w, c] = np.max(a_prev_slice)

    cache = (A_prev, hparameters)

    return A, cache

def conv_backward(dZ, cache):
    '''
    Backpropagation for a convolutional function

    Inputs:
    dZ: np.ndarray -> Gradient of the loss function respect to the output of the conv layer, shape of (m, nH, nW, nC)
    cache: tuple that forward prop function returned

    Returns:
    dA_prev: np.ndarray -> Gradient of the loss respect to the previous input of the conv layers, shape of (m, nH_prev, nW_prev, nC_prev)
    dW: np.ndarray -> Gradient respect to the weights of the conv layers, shape of (f, f, nC_prev, nC)
    db: np.ndarrau -> Gradient respect to the biases of the conv layers, shape of (1, 1, 1, nC)
    '''
    return
