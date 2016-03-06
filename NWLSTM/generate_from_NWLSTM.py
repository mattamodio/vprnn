#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, csv, itertools, operator, os, time, ast
import numpy as np
from datetime import datetime
from utils import *

DATAFILE = "saved_model_parameters/NWLSTM_savedparameters_22069.0__03-06___14-12-16.npz"
SAMPLE_FROM_DISTRIBUTION = True
SAMPLE_LIMIT = 300
SOFTMAX_TEMPERATURES = np.linspace(.1, 1.1, 4)
NUM_SENTENCES = 1

STARTING_STRING = ''' CHAPTER I
    "Well, '''

# import theano
# theano.config.optimizer = 'None'
# theano.config.compute_test_value = 'raise'
# theano.config.exception_verbosity = 'high'
# # #theano.config.profile = True
# theano.config.warn_float64 = 'warn'
# theano.config.floatX = 'float32'

def one_hot(x, dimensions):
    tmp = np.zeros(dimensions).astype('float32')
    tmp[x] = 1
    return tmp.reshape(1, dimensions, 1)

def generate_sentence(model, sample_limit=100, sample_from_distribution=True):
    char_to_code_dict, code_to_char_dict, word_dim = model.char_to_code_dict, model.code_to_char_dict, model.word_dim

    # We start the sentence with the start token
    new_sentence = one_hot(char_to_code_dict[STARTING_STRING[0]], word_dim)
    for c in STARTING_STRING[1:]:
        c_one_hot = one_hot(char_to_code_dict[c], word_dim)
        new_sentence = np.vstack((new_sentence,c_one_hot))

    # Repeat until we get an end token or reach the sample limit
    sampled_letter = None
    while len(new_sentence)<sample_limit:

        next_char_probs = model.forward_propagation(new_sentence) # get probability of next character
        next_char_probs = [_[0] for _ in next_char_probs[-1][:]]        
        
        if sample_from_distribution: # either sample from the distribution or take the most likely next character
            samples = np.random.multinomial(1, next_char_probs)
            sampled_letter = np.argmax(samples)
        else:
            sampled_letter = np.argmax(next_char_probs)

        if sampled_letter not in code_to_char_dict: continue

        sampled_one_hot = one_hot(sampled_letter, word_dim) # convert code to one-hot
        new_sentence = np.vstack((new_sentence,sampled_one_hot)) # stack this one-hot onto sentence thus far


        #if len(new_sentence)%25==0: print len(new_sentence) # print updates every 25 chars
        #if len(new_sentence)%100==0: print "\n" + "".join([code_to_char_dict[np.argmax(x)] for x in new_sentence]) + "\n" # print the sentence so far every 100 chars


    sentence_str = [code_to_char_dict[np.argmax(x)] for x in new_sentence]

    return sentence_str



if __name__=="__main__":
    for softmax_temp in SOFTMAX_TEMPERATURES:
        print "\n\nSoftmax temperature: {0}".format(softmax_temp)
        model = load_model_parameters_lstm(path=DATAFILE,
                                            sample=1,
                                            softmax_temperature=softmax_temp)

        for i in range(NUM_SENTENCES):
            sent = []
            sent = generate_sentence(model, sample_limit=SAMPLE_LIMIT, sample_from_distribution=SAMPLE_FROM_DISTRIBUTION)
            print "".join(sent)
            print


