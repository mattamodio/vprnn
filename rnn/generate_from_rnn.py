import sys
import csv
import itertools
import operator
import numpy as np
import nltk
import os
import time
import ast
import math
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
from utils import load_model_parameters_theano, save_model_parameters_theano
import theano.tensor as T


DATAFILE = "rnn-theano-100-82-2015-12-13-18-21-22.npz"
MODEL = RNNTheano(int(DATAFILE.split('-')[3]), hidden_dim=int(DATAFILE.split('-')[2]))





line_start_token = "LINE_START"
line_end_token = "LINE_END"
dictFile = 'dictFile.txt'
with open(dictFile) as f:
    dicts = []
    for line in f:
        line = ast.literal_eval(line)
        dicts.append(line)
char_to_code_dict, code_to_char_dict = dicts

load_model_parameters_theano('saved_model_parameters/{0}'.format(DATAFILE), MODEL)

def one_hot(x):
    oneHot = np.zeros(82)
    oneHot[x] = 1
    return oneHot

e = np.array([math.e for _ in xrange(82)])
def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [one_hot(char_to_code_dict[line_start_token])]
    # Repeat until we get an end token
    sampled_letter = None
    counter=0
    while sampled_letter != char_to_code_dict[line_end_token]:
        counter+=1
        if counter>15:
            break
        next_char_probs = model.forward_propagation(new_sentence)

        powered = np.power(e, next_char_probs[-1])
        summed = np.sum(powered)
        normalized_probs = np.multiply(powered, 1./summed)


        samples = np.random.multinomial(1, normalized_probs)
        sampled_letter = np.argmax(samples)

        try:
            code_to_char_dict[sampled_letter]
        except:
            continue

        new_sentence.append(one_hot(sampled_letter))
    sentence_str = [code_to_char_dict[np.argmax(x)] for x in new_sentence[1:-1]]
    return sentence_str

num_sentences = 10
 
for i in range(num_sentences):
    sent = []
    sent = generate_sentence(MODEL)
    print "".join(sent)