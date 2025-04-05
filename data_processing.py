import numpy as np
import gzip
from PIL import Image

def load_mnist(train_images_path, train_labels_path, test_images_path, test_labels_path):
    with gzip.open(train_images_path, 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(train_labels_path, 'rb')as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(test_images_path, 'rb') as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(test_labels_path, 'rb')as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    # Reshape tensors
    X_train, X_test = train_images.T, test_images.T
    Y_train, Y_test = train_labels.reshape(1, -1), test_labels.reshape(1, -1)

    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = load_mnist('/Users/Admin/Documents/MachineLearning/datasets/data/MNIST/raw/train-images-idx3-ubyte.gz', 
                                        '/Users/Admin/Documents/MachineLearning/datasets/data/MNIST/raw/train-labels-idx1-ubyte.gz',
                                        '/Users/Admin/Documents/MachineLearning/datasets/data/MNIST/raw/t10k-images-idx3-ubyte.gz',
                                        '/Users/Admin/Documents/MachineLearning/datasets/data/MNIST/raw/t10k-labels-idx1-ubyte.gz')

print('Training shapes:')
print(X_train.shape)
print(Y_train.shape)

print('Test shapes:')
print(X_test.shape)
print(Y_test.shape)
