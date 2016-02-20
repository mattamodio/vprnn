import numpy as np
import sys
from rmsprop_lstm_with_stack import LSTM_with_stack

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)


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
        W_hy=model.W_hy.get_value(),
        #W_h_push=model.W_h_push.get_value(),
        W_h_prev_pop=model.W_h_prev_pop.get_value(),
        W_h_stack_pop=model.W_h_stack_pop.get_value(),
        W_x_i_2=model.W_x_i_2.get_value(),
        W_h_i_2=model.W_h_i_2.get_value(),
        W_x_o_2=model.W_x_o_2.get_value(),
        W_h_o_2=model.W_h_o_2.get_value(),
        W_x_f_2=model.W_x_f_2.get_value(),
        W_h_f_2=model.W_h_f_2.get_value(),
        W_x_g_2=model.W_x_g_2.get_value(),
        W_h_g_2=model.W_h_g_2.get_value())
    print "Saved model parameters to %s." % outfile

def load_model_parameters_lstm(path, minibatch_size=1, push_vec=None, pop_vec=None):
    npzfile = np.load(path)
    W_x_i, W_h_i = npzfile["W_x_i"], npzfile["W_h_i"]
    W_x_o, W_h_o = npzfile["W_x_o"], npzfile["W_h_o"]
    W_x_f, W_h_f = npzfile["W_x_f"], npzfile["W_h_f"]
    W_x_g, W_h_g = npzfile["W_x_g"], npzfile["W_h_g"]
    W_hy = npzfile["W_hy"]
    W_h_prev_pop, W_h_stack_pop = npzfile["W_h_prev_pop"], npzfile["W_h_stack_pop"]
    W_x_i_2, W_h_i_2 = npzfile["W_x_i_2"], npzfile["W_h_i_2"]
    W_x_o_2, W_h_o_2 = npzfile["W_x_o_2"], npzfile["W_h_o_2"]
    W_x_f_2, W_h_f_2 = npzfile["W_x_f_2"], npzfile["W_h_f_2"]
    W_x_g_2, W_h_g_2 = npzfile["W_x_g_2"], npzfile["W_h_g_2"]

    #W_h_push =n pzfile["W_h_push"]
    hidden_dim, word_dim = W_x_i.shape[0], W_x_i.shape[1]
    print "Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
    sys.stdout.flush()
    if push_vec != None:
        model = LSTM_with_stack(word_dim, hidden_dim=hidden_dim, minibatch_size=minibatch_size, push_vec=push_vec, pop_vec=pop_vec)
    else:
        model = LSTM_with_stack(word_dim, hidden_dim=hidden_dim, minibatch_size=minibatch_size)
    model.W_x_i.set_value(W_x_i)
    model.W_h_i.set_value(W_h_i)
    model.W_x_o.set_value(W_x_o)
    model.W_h_o.set_value(W_h_o)
    model.W_x_f.set_value(W_x_f)
    model.W_h_f.set_value(W_h_f)
    model.W_x_g.set_value(W_x_g)
    model.W_h_g.set_value(W_h_g)
    model.W_hy.set_value(W_hy)
    #model.W_h_push.set_value(W_h_push)
    model.W_h_stack_pop.set_value(W_h_stack_pop)
    model.W_h_prev_pop.set_value(W_h_prev_pop)
    model.W_x_i_2.set_value(W_x_i_2)
    model.W_h_i_2.set_value(W_h_i_2)
    model.W_x_o_2.set_value(W_x_o_2)
    model.W_h_o_2.set_value(W_h_o_2)
    model.W_x_f_2.set_value(W_x_f_2)
    model.W_h_f_2.set_value(W_h_f_2)
    model.W_x_g_2.set_value(W_x_g_2)
    model.W_h_g_2.set_value(W_h_g_2)
    
    return model
