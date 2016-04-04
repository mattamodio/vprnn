import numpy as np
import sys,time
from LSTM_Net import LSTM_Net

def getCharDicts():
    NULL = 'NULL'
    chars = [NULL, ' ', '\t', '\r', '\n'] + [chr(x) for x in xrange(33,127)]
    code_to_char_dict = dict([(i,c) for i,c in enumerate(sorted(chars))])
    char_to_code_dict = dict([(code_to_char_dict[k],k) for k in code_to_char_dict])

    ALPHABET_LENGTH = len(code_to_char_dict.keys())

    return char_to_code_dict, code_to_char_dict, ALPHABET_LENGTH

def save_model_parameters_lstm(outfile, model):
    values = dict([(str(p),p.get_value()) for p in model.params])
    values['hidden_dim'] = model.hidden_dim
    values['word_dim'] = model.word_dim
    values['num_layers'] = len(model.layers)
    values['minibatch_dim'] = model.minibatch_dim
    values['word_dim'] = model.word_dim

    np.savez(outfile, **values)
    print "Saved {0} model parameters to {1}".format(len(values.keys()), outfile)

def load_model_parameters_lstm(path, softmax_temperature=1, activation='tanh', stack_height=15):
    npzfile = np.load(path)

    print "Building model from {0} with hidden_dim: {1}, word_dim: {2} and num_layers: {3}".format(path, npzfile['hidden_dim'], npzfile['word_dim'], npzfile['num_layers'])
    sys.stdout.flush()

    t1 = time.time()
    model = LSTM_Net(word_dim=npzfile['word_dim'],
        hidden_dim=npzfile['hidden_dim'],
        minibatch_dim=npzfile['minibatch_dim'],
        num_layers=npzfile['num_layers'], 
        activation=activation)

    model.char_to_code_dict, model.code_to_char_dict, _ = getCharDicts()

    for i in xrange(len(model.params)):
        param = model.params[i]
        value = npzfile[str(param)]
        param.set_value(value)

    t2 = time.time()
    print "Building model took {0:.0f} seconds\n".format(t2-t1)
    sys.stdout.flush()

    return model
