import pandas as pd
import numpy as np


DEFAULT_INIT_VALUE = 1


def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


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
        return np.array([1 if i >= 0 else self.alpha for i in x])

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
        else:
            _init_value = DEFAULT_INIT_VALUE

        weights = np.array([[_init_value] * n_input_units] * n_units)
        self.biases = np.array([_init_value] * n_units)
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
        print("_generate_delta")
        deltas = []
        for idx, weights in enumerate(previous_layer.weights):
            #print("self weights {}:{}".format(idx, [w[idx] for w in self.weights]))
            print("delta_next_layer:{}".format(delta_next_layer))
            print("unactivated_results {}:{}".format(idx, previous_layer.unactivated_results[idx]))
            weight_sum = np.array([float(w[idx]) for w in self.weights])
            print("weight_sum:{}".format(weight_sum))
            print("ffff!{}".format(type(delta_next_layer[0])))
            print("first_product:{}".format(str(weight_sum * delta_next_layer)))
            deltas.append(sum(weight_sum * delta_next_layer) * self.activation.derivative(previous_layer.unactivated_results[idx]))

        #if len(deltas) == 1:
        #    deltas = deltas[0]
        print("generated delta:{}".format(deltas))
        return np.array(deltas)

    def update_weights(self, delta_origin, lr, previous_layer):
        # update delta of current layer units
        if previous_layer is not None:
            #previous_layer.deltas = self._generate_delta(delta_origin, previous_layer)
            previous_layer.deltas = self._generate_delta(self.deltas, previous_layer)

        # update weights of current layer units connecting with previous layer units
        for idx, unit_weights in enumerate(self.weights):
            print("updating weights at unit {}".format(idx))
            Z = self.forward_output[idx]
            print("Z:{}".format(Z))
            print("self.deltas :{}".format(self.deltas))
            print("unit {} weights before update:{}".format(idx, self.weights[idx]))
            for i, w in enumerate(unit_weights):
                self_delta = self.deltas[0] if (self.is_output and self.is_output is True) else self.deltas[idx]
                print("self_delta:{}".format(self_delta))
                self.weights[idx][i] = self.weights[idx][i] - lr * Z * self_delta
            print("unit {} weights after update:{}".format(idx, self.weights[idx]))




    def print_layer(self):
        print("=====================================")
        print("Layer | {} input:{}, output:{}, {} units".format(self.activation, self.n_input_units, self.n_units, self.n_total_weights))
        print("=====================================")


class MLP:

    def __init__(self, n_layers, n_units, input_array, init='Ones', lr=0.01):
        self.n_layers = n_layers
        self.n_units = n_units
        self.init = init
        self.lr = lr
        self.input_shape = input_array.shape
        self._construct_layers()

    def _construct_layers(self):
        self.input_layer = Layer(self.n_units, input_array.shape[1], init=self.init)
        self.layers = [self.input_layer]
        self.hidden_layers = []

        #  HIDDEN LAYERS
        _previous_layer = self.input_layer
        for _ in range(self.n_layers):
            hidden_layer = Layer(self.n_units, _previous_layer.output_shape()[1], self.init)
            self.layers.append(hidden_layer)
            self.hidden_layers.append(hidden_layer)
            _previous_layer = hidden_layer

        # OUTPUT LAYER
        output_layer = Layer(1, _previous_layer.output_shape()[1], self.init, activation=Activation('sigmoid'))
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
        print("final_loss:{}".format(final_loss))
        self.output_layer.set_delta(final_loss)
        next_layer = self.output_layer
        reversed_layers = list(reversed(self.layers))
        for idx, current_layer in enumerate(reversed_layers):
            print("---------------------------------")
            print("updating last layer {}, {} units".format(idx, current_layer.n_units))
            if idx == len(reversed_layers) - 1:
                previous_layer = None
            else:
                previous_layer = reversed_layers[idx + 1]
            original_delta = final_loss if idx == 0 else next_layer.deltas
            current_layer.update_weights(original_delta, self.lr, previous_layer)
            next_layer = current_layer
            print("---------------------------------")


    def _chunk_input(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i: i + n]

    def fit(self, x, y, batch_size=32, epochs=1):
        if self.optimizer is None or self.loss is None or self.metrics is None:
            print("Please compile model first.")
            return
        if len(x) != len(y):
            print("x and y are not in the same size!")
            return

        x_chunks = np.array(list(self._chunk_input(x, batch_size)))
        y_chunks = np.array(list(self._chunk_input(y, batch_size)))
        #for x_chunk, y_chunk in zip(x_chunks, y_chunks):

        # FORWARD PROPAGATION
        _current_input = x[0]
        print("_current_input in fit:{}".format(_current_input))
        layer_output = [y[0]]
        #_current_input = x_chunk
        #layer_output = y_chunk
        for layer in self.layers:
            layer_output = layer.output(_current_input)
            _current_input = layer_output


        # COMPUTING LOSS
        loss = np.array([self._compute_loss(layer_output, y[0])])
        #loss = self._compute_loss(layer_output, [y[0]])

        print("layer_output:{}, loss:{}".format(layer_output, loss))

        # BACK PROPAGATION
        self._update_weights(loss)



    def summary(self):
        n_weights_total = 0
        for layer in self.layers:
            n_weights_total += layer.n_total_weights
            layer.print_layer()
        print("{} units in total".format(n_weights_total))


input_array = np.array([[1, 2, 3], [4, 5, 6]])
x, y = input_array, [0, 1]
test = MLP(2, 5, input_array)
print(test.summary())
test.compile('sgd', 'binary_crossentropy', 'accuracy')
test.fit(x, y)

#aa = Activation('sigmoid', alpha=0.01)
#print(aa.activate(-0.88))
