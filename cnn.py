import numpy as np

# /----------------------------------------------------------/
# Activation functions
def sigmoid(x):
    return 1. /(1 + np.exp(-x))

def sigmoid_derivative(values):
    return values * (1 - values)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(values):
    return np.where(values > 0, 1, 0)

def softmax(x):
    e_z = np.exp(x - np.max(x))
    return e_z / e_z.sum()

# /----------------------------------------------------------/
# Padding for inputs

def zero_pad(X, pad):
    '''
    Apply padding to an input volume

    Inputs:
    X: numpy.ndarray of shape (m, nH, nW, nC)
    pad: int, padding size

    Returns:
    X padded
    '''
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0,0))

    return X_pad

class ConvLayer:
    def __init__(self, filters, kernel_size, stride, pad):
        self.nC = filters
        self.f = kernel_size
        self.stride = stride
        self.pad = pad
        self.W = None
        self.b = None
        self.conv_cache = None

    def forward(self, A_prev):
        '''
        Forward propagation for a conv layer

        Inputs:
        A_prev: np.ndarray -> Activation output from the previous layer of shape (m, nH, nW, nC)
        W: np.ndarray -> Filters of shape (f, f, nC_prev, nC) 
        b: np.ndarray -> Biases from the current layer of shape (1, 1, 1, nC)
        hparameters: Dictionary with stride and padding values

        Returns:
        Z: np.ndarray of shape (m, nH, nW, nC)
        cache: tuple -> Saves (A_prev, W, b, hparameters) for backpropagation
        '''
        if self.W is None:
            nC_prev = A_prev.shape[-1]
            self.W = np.random.randn(self.f, self.f, nC_prev, self.nC) * np.sqrt(2. / (self.f * self.f * nC_prev))
            self.b = np.zeros((1, 1, 1, self.nC))

        m, nH_prev, nW_prev, nC_prev = A_prev.shape
        f_h, f_w = self.f, self.f

        nH = int((nH_prev - self.f + 2 * self.pad)/self.stride) + 1
        nW = int((nW_prev - self.f + 2 * self.pad)/self.stride) + 1

        Z = np.zeros((m, nH, nW, self.nC))
        A_prev_pad = zero_pad(A_prev, self.pad) if self.pad > 0 else A_prev

        for i in range(m):
            for h in range(nH):
                h_start = h * self.stride
                h_end = h_start + f_h

                for w in range(nW):
                    w_start = w * self.stride
                    w_end = w_start + f_w

                    # Vectorized implementation
                    a_slice = A_prev_pad[i, h_start:h_end, w_start:w_end, :]
                    Z[i, h, w, :] = np.sum(a_slice[:,:,:,np.newaxis] * self.W, axis=(0,1,2))+ self.b[0,0,0,:]

        self.conv_cache = (A_prev, self.W, self.b, {'stride': self.stride, 'pad': self.pad})

        return Z, self.conv_cache
    
    def conv_backward(self, dZ):
        '''
        Backpropagation for a conv layer

        Inputs:
        dZ: np.ndarray -> Gradient of the loss function respect to the output of the conv layer, shape of (m, nH, nW, nC)
        cache: tuple that forward prop function returned

        Returns:
        dA_prev: np.ndarray -> Gradient of the loss respect to the previous input of the conv layers, shape of (m, nH_prev, nW_prev, nC_prev)
        dW: np.ndarray -> Gradient respect to the weights of the conv layers, shape of (f, f, nC_prev, nC)
        db: np.ndarrauy -> Gradient respect to the biases of the conv layers, shape of (1, 1, 1, nC)
        '''
        (A_prev, W, b, hparameters) = self.conv_cache
        stride, pad = hparameters['stride'], hparameters['pad']

        (m, nH_prev, nW_prev, nC_prev) = A_prev.shape
        (f, f, nC_prev, nC) = W.shape
        (m, nH, nW, nC) = dZ.shape

        dA_prev = np.zeros((m, nH_prev, nW_prev, nC_prev)) 
        dW = np.zeros((W.shape))
        db = np.zeros((b.shape))

        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = np.zeros_like(A_prev_pad)

        for i in range(m):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]

            for h in range(nH):
                for w in range(nW):
                    for c in range(nC):
                        vertical_start = h * stride
                        vertical_end = vertical_start + f

                        horizontal_start = w * stride
                        horizontal_end = horizontal_start + f

                        a_slice = a_prev_pad[vertical_start:vertical_end, horizontal_start:horizontal_end, :]

                        da_prev_pad[vertical_start:vertical_end, horizontal_start:horizontal_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]

                        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :] if pad else da_prev_pad

        dW /= m
        db /= m

        return dA_prev, dW, db

class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        self.f = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride
        self.pool_cache = None

    def forward(self, A_prev):
        '''
        Forward prop for a max pooling layer

        Inputs:
        A_prev: np.ndarray -> Input volume of shape (m, nH_prev, nW_prev, nC_prev)
        hparameters: dictionary containing padding and stride values

        Returns:
        A = np.ndarray -> Output of the pool layer of shape (m, nH, nW, nC)
        cache = tuple for backpropagation in pooling layer
        '''
        (m , nH_prev, nW_prev, nC_prev) = A_prev.shape
        f_w, f_h = self.f    

        # Set output's dimension
        nH = int((nH_prev - f_h) / self.stride) + 1
        nW = int((nW_prev - f_w) / self.stride) + 1

        A = np.zeros((m, nH, nW, nC_prev))

        for i in range(m):
            for h in range(nH):
                h_start = h * self.stride
                h_end = h_start + f_h

                for w in range(nW):
                    w_start = w * self.stride
                    w_end = w_start + f_w

                    for c in range(nC_prev):
                        a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                        A[i, h, w, c] = np.max(a_prev_slice)

        self.pool_cache = (A_prev, {'f' : self.f, 'stride' : self.stride})

        return A, self.pool_cache
    
    def backward(self, dA):
        '''
        Backpropagation for a max pooling layer

        Inputs:
        dA: np.ndarray -> Gradient of the loss function with respect of the output of the pooling layer. Same shape as A
        cache: tuple -> Cache output of the pooling layer, contains layer's and hparameters

        Returns:
        dA_prev -> Gradient of the loss with respect of the input of the input of the pooling layer. Same shape as dA_prev
        '''
        (A_prev, hparameters) = self.pool_cache

        stride = hparameters['stride']
        f_h, f_w = hparameters['f']

        m, nH_prev, nW_prev, nC_prev = A_prev.shape
        m, nH, nW, nC = dA.shape

        dA_prev = np.zeros((A_prev.shape))

        for i in range(m):
            a_prev = A_prev[i]

            for h in range(nH):
                for w in range(nW):
                    for c in range(nC):

                        h_start = h * stride
                        h_end = h_start + f_h

                        w_start = w * stride
                        w_end = w_start + f_w

                        a_prev_slice = a_prev[h_start:h_end, w_start:w_end, c]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[i, h_start:h_end, w_start:w_end, c] += mask * dA[i, h, w, c]

        return dA_prev
    
class FCLayer:
    def __init__(self, input_size, output_size, activation):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2./input_size)
        self.b = np.zeros((1, output_size))
        self.activation = activation
        self.cache = None 

    def linear_forward(self, A):
        Z = np.dot(self.W, A) + self.b
        self.cache = (A, self.W, self.b)

        return Z, self.cache

    def forward_activation(self, A_prev, activation):
        Z, linear_cache = self.linear_forward(A_prev, self.W, self.b)

        if self.activation == 'relu':
            A = relu(Z)
        elif self.activation == 'softmax':
            A = softmax(Z)
        elif self.activation == 'sigmoid':
            A = sigmoid(Z)
        
        activation_cache = Z
        self.cache = (linear_cache, activation_cache)

        return A, self.cache
    
    def forward(self, X):
        pass


