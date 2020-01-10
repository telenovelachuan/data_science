import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivation(x):
    f = 1 / (1 + np.exp(-x))
    return f * (1 - f)


DERIVATIVES = {
    'sigmoid': (sigmoid, sigmoid_derivation)
}

#print(sigmoid(1))
#print(sigmoid_derivation(1))

