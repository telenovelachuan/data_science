import pandas as pd
import numpy as np
import math

DEFAULT_INIT_VALUE = 1


class Layer:

    def __init__(self, n_units, n_input_units, init='Zeros', activation=None):
        self.n_units = n_units
        self.init = init
        self.activation = activation
        self.n_input_units = n_input_units
        self.weights = self._init_units(n_units, self.n_input_units)
        self.n_total_weights = self.n_units * self.n_input_units + len(self.biases)

    def _init_units(self, n_units, n_input_units):

        if self.init == 'Ones':
            _init_value = 1
        else:
            _init_value = DEFAULT_INIT_VALUE

        weights = [[_init_value] * n_input_units] * n_units
        self.biases = np.array([_init_value] * n_units)
        return weights

    def output_shape(self):
        return 1, self.n_units

    def output(self, input_array):
        result = np.array([sum(np.array(w) * input_array) for w in self.weights])
        if self.activation == 'sigmoid':
            result = 1/(1 + np.exp(-result))
        elif self.activation == 'Leaky ReLU':
            result = np.where(result > 0, result, result * 0.01)
        return result

    def print_layer(self):
        print("=====================================")
        print("Layer | input:{}, output:{}, {} units".format(self.n_input_units, self.n_units, self.n_total_weights))
        print("=====================================")


class MLP:

    def __init__(self, n_layers, n_units, input_array, init='Ones'):
        self.n_layers = n_layers
        self.n_units = n_units
        self.init = init
        self.input_shape = input_array.shape
        self._construct_layers()

    def _construct_layers(self):
        self.input_layer = Layer(self.n_units, input_array.shape[1], init=self.init)
        self.layers = [self.input_layer]

        #  HIDDEN LAYERS
        _previous_layer = self.input_layer
        for _ in range(self.n_layers):
            hidden_layer = Layer(self.n_units, _previous_layer.output_shape()[1], self.init)
            self.layers.append(hidden_layer)
            _previous_layer = hidden_layer

        # OUTPUT LAYER
        self.layers.append(Layer(1, _previous_layer.output_shape()[1], self.init))

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def fit(self, x, y, batch_size=None, epochs=1):
        

    def summary(self):
        n_weights_total = 0
        for layer in self.layers:
            n_weights_total += layer.n_total_weights
            layer.print_layer()
        print("{} units in total".format(n_weights_total))


input_array = np.array([[1, 2, 3], [4, 5, 6]])
test = MLP(2, 5, input_array)
test.summary()
