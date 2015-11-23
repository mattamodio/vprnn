import numpy as np
import sys
import os


class RNN(object):

    def __init__(self, layers):
        """Initializes a neural net with the layers whose sizes are in 'layers'. The first item in
        the list is the input layer, and the last is the output layer. Weights and biases randomly
        initialized to a random normal(0,1) variable."""
        self.num_layers = len(layers)
        self.layers = layers
        # one hidden layer to start
        self.W_xh, self.W_hy = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.W_hh = np.random.randn(layers[1], layers[1])

        self.h = np.random.randn(layers[1], 1)

    def step(self, input_x):
        if not hasattr(input_x, 'shape') or input_x.shape != (self.layers[0],):
            input_x = self.encode(input_x)
        # updated hidden state
        self.h = sigmoid(np.dot(self.W_xh, input_x) + np.dot(self.W_hh, self.h))

        # compute output
        return self.decode( np.dot(self.W_hy, self.h) )

    def __iter__(self):
        """"""
        def iterHelper():
            y = 1
            while True:
                y = self.step(y)
                yield y
        return iterHelper()

    def decode(self, y):

        return np.argmax(y)

    def encode(self, x):
        encoded = np.zeros((self.layers[0],1))
        encoded[x] = 1
        return encoded

def sigmoid(z):
    """Sigmoid function evaluated at z."""
    return 1.0/(1.0+np.exp(-z))


def main():
    layers = [5,5,5]
    rnn = RNN(layers)

    counter = 0
    for i in rnn:
        print i
        counter +=1
        if counter>3:
            break





if __name__ == "__main__":
    main()