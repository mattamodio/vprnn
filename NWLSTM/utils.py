import numpy as np
import sys,time
from NWLSTM_Net import NWLSTM_Net



def save_model_parameters_lstm(outfile, model):
    values = dict([(str(p),p.get_value()) for p in model.params])
    values['hidden_dim'] = model.hidden_dim
    values['word_dim'] = model.word_dim
    values['num_layers'] = len(model.layers)
    values['want_stack'] = model.want_stack
    values['minibatch_dim'] = model.minibatch_dim
    values['word_dim'] = model.word_dim
    values['PUSH'] = model.PUSH.eval()
    values['POP'] = model.POP.eval()
    values['char_to_code_dict'] = model.char_to_code_dict
    values['code_to_char_dict'] = model.code_to_char_dict

    np.savez(outfile, **values)
    print "Saved {0} model parameters to {1}".format(len(values.keys()), outfile)

def load_model_parameters_lstm(path, sample=1, softmax_temperature=1):
    npzfile = np.load(path)

    print "Building model from {0} with hidden_dim: {1}, word_dim: {2} and num_layers: {3}".format(path, npzfile['hidden_dim'], npzfile['word_dim'], npzfile['num_layers'])
    sys.stdout.flush()

    t1 = time.time()
    model = NWLSTM_Net(word_dim=npzfile['word_dim'],
        hidden_dim=npzfile['hidden_dim'],
        minibatch_dim=sample,
        num_layers=npzfile['num_layers'], 
        activation='tanh',
        want_stack=npzfile['want_stack'],
        stack_height=15, 
        push_vec=npzfile['PUSH'],
        pop_vec=npzfile['POP'])

    model.char_to_code_dict = npzfile['char_to_code_dict'].item()
    model.code_to_char_dict = npzfile['code_to_char_dict'].item()

    for i in xrange(len(model.params)):
        param = model.params[i]
        value = npzfile[str(param)]
        param.set_value(value)

    t2 = time.time()
    print "Building model took {0:.0f} seconds\n".format(t2-t1)
    sys.stdout.flush()

    return model
