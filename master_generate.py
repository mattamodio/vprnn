import sys, csv, itertools, operator, os, time, ast
import numpy as np
from datetime import datetime
from master_utils import *
from master_class import LSTMTheano

DATAFILE = "lstm_rmsprop-theano-1024-83-2016-03-02-01-07-32.npz"
dictFile = 'dictFile.txt'
sample_from_distribution = True
SAMPLE_LIMIT = 299

STARTING_STRING = '''<game atBat="431145"'''


def one_hot(x, dimensions):
    tmp = np.zeros(dimensions).astype('int32')
    tmp[x] = 1
    return tmp.reshape(1, dimensions, 1)

def generate_sentence(model, char_to_code_dict, code_to_char_dict, line_start_token, line_end_token, alphabet_length, sample_limit=sys.maxint):
    # We start the sentence with the start token
    new_sentence = one_hot(char_to_code_dict[line_start_token], alphabet_length)
    for c in STARTING_STRING:
        c_one_hot = one_hot(char_to_code_dict[c], alphabet_length)
        new_sentence = np.vstack((new_sentence,c_one_hot))
    # Repeat until we get an end token
    sampled_letter = None
    while len(new_sentence)<sample_limit:

        next_char_probs = model.forward_propagation(new_sentence)
        next_char_probs = [_[0] for _ in next_char_probs[-1][:]]

        
        if sample_from_distribution:
            samples = np.random.multinomial(1, next_char_probs)
            sampled_letter = np.argmax(samples)
        else:
            sampled_letter = np.argmax(next_char_probs)

        sampled_one_hot = one_hot(sampled_letter, alphabet_length)

        new_sentence = np.vstack((new_sentence,sampled_one_hot))

        if len(new_sentence)%25==0: print len(new_sentence)
        if len(new_sentence)%100==0: print "\n" + "".join([code_to_char_dict[np.argmax(x)] for x in new_sentence]) + "\n"

        if code_to_char_dict[sampled_letter] == line_end_token: break

    sentence_str = [code_to_char_dict[np.argmax(x)] for x in new_sentence]

    return sentence_str



if __name__=="__main__":
    
    
    with open(dictFile) as f:
        dicts = []
        for line in f:
            line = ast.literal_eval(line)
            dicts.append(line)
    char_to_code_dict, code_to_char_dict = dicts
    line_start_token = 'LINE_START'
    line_end_token = 'LINE_END'
    ALPHABET_LENGTH = int(DATAFILE.split('-')[3])


    push_vec = one_hot(char_to_code_dict['<'], ALPHABET_LENGTH)
    pop_vec = one_hot(char_to_code_dict['>'], ALPHABET_LENGTH)

    MODEL = load_model_parameters_lstm('saved_model_parameters/{0}'.format(DATAFILE), minibatch_dim=1, push_vec=push_vec, pop_vec=pop_vec)



    num_sentences = 1
     
    for i in range(num_sentences):
        sent = []
        sent = generate_sentence(MODEL, char_to_code_dict, code_to_char_dict, line_start_token, line_end_token, ALPHABET_LENGTH, sample_limit=SAMPLE_LIMIT)
        print "".join(sent)
        print


