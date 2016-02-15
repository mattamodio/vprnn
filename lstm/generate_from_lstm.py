import sys, csv, itertools, operator, os, time, ast
import numpy as np
from datetime import datetime
from utils import *
from lstm_theano import LSTMTheano

DATAFILE = "lstm-theano-500-82-2016-02-14-18-06-09.npz"

def one_hot(x, dimensions):
    tmp = np.zeros(dimensions).astype('int32')
    tmp[x] = 1
    return tmp.reshape(1, dimensions, 1)

def generate_sentence(model, char_to_code_dict, code_to_char_dict, line_start_token, line_end_token, alphabet_length, sample_limit=sys.maxint):
    # We start the sentence with the start token
    new_sentence = one_hot(char_to_code_dict[line_start_token], alphabet_length)
    # Repeat until we get an end token
    sampled_letter = None
    while len(new_sentence)<sample_limit:

        next_char_probs = model.forward_propagation(new_sentence)

        next_char_probs = [_[0] for _ in next_char_probs[-1][:]]

        samples = np.random.multinomial(1, next_char_probs)
        sampled_letter = np.argmax(samples)
        sampled_one_hot = one_hot(sampled_letter, alphabet_length)

        # try:
        #     code_to_char_dict[sampled_letter]
        # except:
        #     continue

        new_sentence = np.vstack((new_sentence,sampled_one_hot))

        if code_to_char_dict[sampled_letter] == line_end_token: break

    sentence_str = [code_to_char_dict[np.argmax(x)] for x in new_sentence]

    return sentence_str



if __name__=="__main__":
    
    dictFile = 'dictFile.txt'
    with open(dictFile) as f:
        dicts = []
        for line in f:
            line = ast.literal_eval(line)
            dicts.append(line)
    char_to_code_dict, code_to_char_dict = dicts
    line_start_token = 'LINE_START'
    line_end_token = 'LINE_END'
    ALPHABET_LENGTH = 82

    MODEL = load_model_parameters_lstm('saved_model_parameters/{0}'.format(DATAFILE), minibatch_size=1)



    num_sentences = 5
     
    for i in range(num_sentences):
        sent = []
        sent = generate_sentence(MODEL, char_to_code_dict, code_to_char_dict, line_start_token, line_end_token, ALPHABET_LENGTH, sample_limit=300)
        print "".join(sent)
        print


