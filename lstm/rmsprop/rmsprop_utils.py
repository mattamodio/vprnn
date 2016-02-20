import numpy as np
import sys
from rmsprop_lstm_theano import LSTMTheano


def save_model_parameters_lstm(outfile, model):
    np.savez(outfile,
        W_x_i=model.W_x_i.get_value(),
        W_h_i=model.W_h_i.get_value(),
        W_x_o=model.W_x_o.get_value(),
        W_h_o=model.W_h_o.get_value(),
        W_x_f=model.W_x_f.get_value(),
        W_h_f=model.W_h_f.get_value(),
        W_x_g=model.W_x_g.get_value(),
        W_h_g=model.W_h_g.get_value(),
        W_hy=model.W_hy.get_value())
    print "Saved model parameters to %s." % outfile

def load_model_parameters_lstm(path, minibatch_size=1):
    npzfile = np.load(path)
    W_x_i, W_h_i = npzfile["W_x_i"], npzfile["W_h_i"]
    W_x_o, W_h_o = npzfile["W_x_o"], npzfile["W_h_o"]
    W_x_f, W_h_f = npzfile["W_x_f"], npzfile["W_h_f"]
    W_x_g, W_h_g = npzfile["W_x_g"], npzfile["W_h_g"]
    W_hy = npzfile["W_hy"]
    hidden_dim, word_dim = W_x_i.shape[0], W_x_i.shape[1]
    print "Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
    sys.stdout.flush()
    model = LSTMTheano(word_dim, hidden_dim=hidden_dim, minibatch_size=minibatch_size)
    model.W_x_i.set_value(W_x_i)
    model.W_h_i.set_value(W_h_i)
    model.W_x_o.set_value(W_x_o)
    model.W_h_o.set_value(W_h_o)
    model.W_x_f.set_value(W_x_f)
    model.W_h_f.set_value(W_h_f)
    model.W_x_g.set_value(W_x_g)
    model.W_h_g.set_value(W_h_g)
    model.W_hy.set_value(W_hy)
    return model
