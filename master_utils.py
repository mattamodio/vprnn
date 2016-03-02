import numpy as np
import sys
from master_class import LSTMTheano


def save_model_parameters_lstm(outfile, model):
    #values = [p.get_value() for p in model.params]
    values = {str(p):p.get_value() for p in model.params}
    values['hidden_dim'] = model.hidden_dim
    values['word_dim'] = model.word_dim
    values['num_layers'] = len(model.layers)
    #np.savez(outfile, hidden_dim=model.hidden_dim, word_dim=model.word_dim, num_layers=len(model.layers), *values)
    np.savez(outfile, **values)
    print "Saved model parameters to %s." % outfile

def load_model_parameters_lstm(path, minibatch_dim=1, push_vec=None, pop_vec=None):
    npzfile = np.load(path)
    hidden_dim, word_dim, num_layers = npzfile['hidden_dim'], npzfile['word_dim'], npzfile['num_layers']
    print "Building model model from %s" % (path)
    sys.stdout.flush()

    if push_vec != None:
        model = LSTMTheano(word_dim, hidden_dim=hidden_dim, minibatch_dim=minibatch_dim, num_layers=num_layers, 
            activation='tanh', want_stack=True, stack_height=15, push_vec=push_vec, pop_vec=pop_vec)
    else:
        model = LSTMTheano(word_dim, hidden_dim=hidden_dim, minibatch_dim=minibatch_dim, num_layers=num_layers)

    #model = LSTMTheano(word_dim, hidden_dim=hidden_dim, minibatch_dim=minibatch_dim, num_layers=num_layers)

    for i in xrange(len(model.params)):
        param = model.params[i]
        value = npzfile[str(param)]
        param.set_value(value)
        

    return model
