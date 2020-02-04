import pandas as pd
import numpy as np
import sys
import random
import matplotlib.pyplot as plt


DEFAULT_INIT_VALUE = 1


def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


def mse(predictions, targets):
    if isinstance(predictions, np.ndarray) or isinstance(predictions, list):
        return (np.square(predictions - targets)).mean()
    else:
        return ((predictions - targets)**2).mean()


class Activation:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        f = 1 / (1 + np.exp(-x))
        return f * (1 - f)

    def leaky_relu(self, x):
        return np.where(x > 0, x, x * self.alpha)

    def leaky_relu_derivative(self, x):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            return np.array([1 if i >= 0 else self.alpha * i for i in x])
        else:
            return 1 if x >= 0 else self.alpha * x

    @staticmethod
    def none(x):
        return x

    @staticmethod
    def none_derivative(x):
        return 1

    def __init__(self, name, **kwargs):
        self.name = name
        _predifined_methods = {
            'sigmoid': (self.sigmoid.__get__(None, object), self.sigmoid_derivative.__get__(None, object)),
            'leakyReLU': (self.leaky_relu, self.leaky_relu_derivative),
            'none': (self.none.__get__(None, object), self.none_derivative.__get__(None, object))
        }
        if name not in _predifined_methods:
            print("{} not supported")
            return
        self.activate, self.derivative = _predifined_methods[name]
        self.__dict__.update(kwargs)

    def __str__(self):
        return self.name


class Layer:

    def __init__(self, n_units, n_input_units, init='Zeros', activation=Activation('none')):
        self.n_units = n_units
        self.init = init
        self.activation = activation
        self.n_input_units = n_input_units
        self.weights = self._init_units(n_units, self.n_input_units)
        self.n_total_weights = self.n_units * self.n_input_units + len(self.biases)
        self.is_output = False

    def _init_units(self, n_units, n_input_units):
        if self.init == 'Ones':
            _init_value = 1
            weights = np.array([[_init_value] * n_input_units] * n_units)
            self.biases = np.array([_init_value] * n_units)
        elif isinstance(self.init, float) or isinstance(self.init, int):
            _init_value = self.init
            weights = np.array([[_init_value] * n_input_units] * n_units)
            self.biases = np.array([_init_value] * n_units)
        elif self.init == 'uniform':
            value_pool = np.linspace(-0.5, 0.5, n_input_units * n_units)
            random.shuffle(value_pool)
            weights = np.array([[0.1] * n_input_units] * n_units)
            for idx_u in range(n_units):
                for idx_i in range(n_input_units):
                    weights[idx_u][idx_i] = value_pool[idx_i + idx_u * n_input_units]
            random.shuffle(value_pool)
            self.biases = np.array(value_pool[:n_units])
        else:
            _init_value = DEFAULT_INIT_VALUE
            weights = np.array([[_init_value] * n_input_units] * n_units)
            self.biases = np.array([_init_value] * n_units)
        print("initialized weights:{}".format(weights))
        return weights

    def output_shape(self):
        return 1, self.n_units

    def output(self, input_array):
        self.input_array = np.array(input_array)
        results = []
        for idx, w in enumerate(self.weights):
            product = sum(self.input_array * np.array(w))
            biased = product + self.biases[idx]
            results.append(biased)
        results = np.array(results)
        self.unactivated_results = results

        # result = np.array([sum(np.array(w) * input_array) + self.biases for w in self.weights])
        if self.activation:
            results = self.activation.activate(results)

        self.forward_output = results
        return results

    def set_delta(self, loss):
        self.deltas = loss

    def set_last(self):
        self.is_output = True

    def _generate_delta(self, delta_next_layer, previous_layer):
        #print("_generate_delta")
        deltas = []
        for idx, weights in enumerate(previous_layer.weights):
            weight_sum = np.array([float(w[idx]) for w in self.weights])
            #print("unactivated:{}".format(previous_layer.unactivated_results[idx]))
            #print("sum:{}, derivative:{}".format(sum(weight_sum * delta_next_layer), self.activation.derivative(previous_layer.unactivated_results[idx])))
            deltas.append(sum(weight_sum * delta_next_layer) * self.activation.derivative(previous_layer.unactivated_results[idx]))

        #if len(deltas) == 1:
        #    deltas = deltas[0]
        #print("generated delta:{}".format(deltas))
        return np.array(deltas)

    def update_weights(self, delta_origin, lr, previous_layer):
        # update delta of current layer units
        if previous_layer is not None:
            previous_layer.deltas = self._generate_delta(self.deltas, previous_layer)

        # update weights of current layer units connecting with previous layer units
        for idx, unit_weights in enumerate(self.weights):
            Z = self.forward_output[idx]
            for i, w in enumerate(unit_weights):
                self_delta = self.deltas[0] if (self.is_output and self.is_output is True) else self.deltas[idx]
                #print("lr:{}, Z:{}, self_delta:{}".format(lr, Z, self_delta))
                #print("current weight:{}, - {}".format(self.weights[idx][i], lr * Z * self_delta))
                self.weights[idx][i] = self.weights[idx][i] - lr * Z * self_delta


    def print_layer(self):
        print("=====================================")
        print("Layer | {} input:{}, output:{}, {} units".format(self.activation, self.n_input_units, self.n_units, self.n_total_weights))
        print("=====================================")

    def print_weigts(self):
        print("-------------------------------------")
        print("Layer | {} {}".format(self.activation, self.weights))
        print("-------------------------------------")


class MLP:

    def __init__(self, n_layers, n_units, input_array, init='Ones', lr=0.01):
        self.n_layers = n_layers
        self.n_units = n_units
        self.init = init
        self.lr = lr
        self.input_shape = input_array.shape
        self.input_array = input_array
        self._construct_layers()

    def _construct_layers(self):
        print("input_array :{}".format(self.input_array))
        self.input_layer = Layer(self.n_units, self.input_array.shape[1], init=self.init)
        self.layers = [self.input_layer]
        self.hidden_layers = []

        #  HIDDEN LAYERS
        _previous_layer = self.input_layer
        for _ in range(self.n_layers):
            hidden_layer = Layer(self.n_units, _previous_layer.output_shape()[1], init=self.init)
            self.layers.append(hidden_layer)
            self.hidden_layers.append(hidden_layer)
            _previous_layer = hidden_layer

        # OUTPUT LAYER
        output_layer = Layer(1, _previous_layer.output_shape()[1], init=self.init, activation=Activation('leakyReLU', alpha=0.1))
        output_layer.set_last()
        self.layers.append(output_layer)
        self.output_layer = output_layer

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def _compute_loss(self, prediction, y):
        if self.loss == 'binary_crossentropy':
            loss = cross_entropy(prediction, y)
        else:
            loss = (np.square(prediction - y)).mean()
        return loss

    def _update_weights(self, final_loss):
        #print("final_loss:{}".format(final_loss))
        self.output_layer.set_delta(final_loss)
        next_layer = self.output_layer
        reversed_layers = list(reversed(self.layers))
        for idx, current_layer in enumerate(reversed_layers):
            #print("---------------------------------")
            #print("updating last layer {}, {} units".format(idx, current_layer.n_units))
            if idx == len(reversed_layers) - 1:
                previous_layer = None
            else:
                previous_layer = reversed_layers[idx + 1]
            original_delta = final_loss if idx == 0 else next_layer.deltas
            current_layer.update_weights(original_delta, self.lr, previous_layer)
            next_layer = current_layer
            #print("---------------------------------")

    def _chunk_input(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i: i + n]

    def _compute_output(self, input_array):
        _input = input_array
        for layer in self.layers:
            layer_output = layer.output(_input)
            _input = layer_output
        return layer_output

    def _compute_total_loss(self, x, y):
        predictions = np.array([self._compute_output(row) for row in x])
        return self._compute_loss(predictions, y)

    def fit(self, x, y, batch_size=32, epochs=1):
        if self.optimizer is None or self.loss is None or self.metrics is None:
            print("Please compile model first.")
            return
        if len(x) != len(y):
            print("x and y are not in the same size!")
            return

        x_chunks = np.array(list(self._chunk_input(x, batch_size)))
        y_chunks = np.array(list(self._chunk_input(y, batch_size)))
        zipp = list(zip(x_chunks, y_chunks))
        history = {
            'loss': []
        }
        dots_per_epoch = 30
        for epoch in range(epochs):
            print("\n-----------------------------------------")
            print("Training epoch {} / {}".format(epoch + 1, epochs))
            for chunk_idx, (x_chunk, y_chunk) in enumerate(zipp):
                predictions = []
                ys = []
                for x_single, y_single in zip(x_chunk, y_chunk):
                    # FORWARD PROPAGATION
                    _current_input = x_single
                    #print("_current_input in fit:{}".format(_current_input))
                    current_output = self._compute_output(_current_input)
                    #print("calculated output:{}".format(current_output))

                    predictions.append(current_output)
                    ys.append(y_single)

                # COMPUTING LOSS
                loss = np.array([self._compute_loss(np.array(predictions), np.array(ys))])
                # loss = self._compute_loss(layer_output, [y[0]])
                #print("predictions:{}, losses:{}".format(predictions, loss))

                # BACK PROPAGATION
                self._update_weights(loss)
                total_loss = self._compute_total_loss(x, y)
                history['loss'].append(total_loss)

                dots = int((float(chunk_idx + 1) / len(x_chunks)) * dots_per_epoch)
                unfinished_dots = dots_per_epoch - dots
                progress_str = "{} / {} [{}{}] - Total loss: {}".format(
                    chunk_idx + 1, len(zipp), "".join(["="] * dots), "".join(["."] * unfinished_dots), total_loss)
                sys.stdout.write('\r' + progress_str)

            #print("-----------------------------------------")
        return history

    def summary(self):
        n_weights_total = 0
        for layer in self.layers:
            n_weights_total += layer.n_total_weights
            layer.print_layer()
        print("{} units in total".format(n_weights_total))


#input_array = np.array([[-0.01, -0.02, -0.03], [0.04, 0.05, 0.006]])
#x, y = input_array, np.array([1, 0])

x = pd.read_csv("~/Desktop/x.csv").values
y = pd.read_csv("~/Desktop/y.csv").values

test = MLP(2, 2, x, init='uniform', lr=0.01)
print(test.summary())
test.compile('sgd', 'mse', 'accuracy')
history = test.fit(x, y, epochs=500)

# print(history)
# plt.plot(history['loss'])
#plt.show()

