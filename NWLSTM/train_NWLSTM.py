#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, csv, itertools, operator, os, time, ast
import numpy as np
from datetime import datetime
import theano
import theano.tensor as T
from utils import *
from NWLSTM_Net import NWLSTM_Net

# Data parameters
CHAR_LIMIT = False
MINIBATCH_SIZE = 50
SEQUENCE_LENGTH = 100
BPTT_TRUNCATE = 20
LINE_START_TOKEN = "LINE_START"
LINE_END_TOKEN = "LINE_END"

# Stack parameters
WANT_STACK = False
STACK_HEIGHT = 15
PUSH_CHAR = u'\t'
POP_CHAR = u'\n'

# Layer parameters
HIDDEN_DIM = 512
NUM_LAYERS = 2
ACTIVATION = 'tanh' # tanh or relu

# Optimization parameters
OPTIMIZATION = 'RMSprop' # RMSprop or SGD
LEARNING_RATE = .01
DROPOUT = .5
NEPOCH = 100

# Data source parameters
DATAFILE = '../data/war_and_peace.txt'
MODEL_FILE = None
#MODEL_FILE = 'saved_model_parameters/lstm_rmsprop-theano-500-83-2016-02-19-18-13-55.npz'


theano.config.optimizer = 'None'
# theano.config.compute_test_value = 'raise'
# theano.config.exception_verbosity = 'high'
# #theano.config.profile = True
theano.config.warn_float64 = 'warn'
theano.config.floatX = 'float32'
# theano.config.device = 'gpu'

print "Training NWLSTM with parameters:\
    \nLength of training set: {0}\
    \nHidden Layer: {1}\
    \nNumber of Layers: {2}\
    \nStack: {8}\
    \nLearning Rate: {3}\
    \nDropout: {9}\
    \nMinibatch size: {4}\
    \nSequence Length: {5}\
    \nTruncate gradient:{10}\
    \nNumber of epochs: {6}\
    \nLoading from model: {7}\n".format(CHAR_LIMIT, HIDDEN_DIM, NUM_LAYERS, LEARNING_RATE, MINIBATCH_SIZE,
        SEQUENCE_LENGTH, NEPOCH, MODEL_FILE, WANT_STACK, DROPOUT, BPTT_TRUNCATE)

def one_hot(x, dimensions):
    tmp = np.zeros(dimensions)
    tmp[x] = 1
    return tmp.astype('float32')

def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=1, evaluate_loss_after=1):
    # We keep track of the losses so we can plot them later
    losses = []
    examples_per_epoch = (CHAR_LIMIT - SEQUENCE_LENGTH) / MINIBATCH_SIZE
    percent_progress_points = set([int((i/20.) * examples_per_epoch) for i in range(1,10)])
    print "Will print progress at steps: {0}".format(sorted(list(percent_progress_points)))

    for epoch in range(nepoch):
        num_examples_seen = 0
        percent_progress = 1
        
        if (epoch % evaluate_loss_after == 0): # Optionally evaluate the loss
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = time.strftime("%m-%d___%H-%M-%S")
            print "{0}: Loss after epoch={1}: {2}".format(time, epoch, loss)
            save_model_parameters_lstm("saved_model_parameters/NWLSTM_savedparameters_{0:.1f}__{1}.npz".format(loss, time), model)

        for i in xrange(len(y_train)): # One SGD step
            model.train_model(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
            # if num_examples_seen in percent_progress_points:
            #     print "{0}% finished".format(str(percent_progress)+"0")
            #     percent_progress+=1
            #     sys.stdout.flush()

def readFile(filename, dictsFile=None, outputDictsToFile=None):
    global CHAR_LIMIT
    with open(filename, 'rb') as f:
        char_to_code_dict = {LINE_START_TOKEN: 0, LINE_END_TOKEN: 1}
        code_to_char_dict = {0: LINE_START_TOKEN, 1: LINE_END_TOKEN}
        ALPHABET_LENGTH = 1

        filestring = f.read()
        filestring = filestring.decode('utf-8')
        if not CHAR_LIMIT:
            CHAR_LIMIT = len(filestring)
        c_list = [LINE_START_TOKEN]
        for c in filestring:
            c_list.append(c)

            if c not in char_to_code_dict:
                ALPHABET_LENGTH+=1
                char_to_code_dict[c] = ALPHABET_LENGTH
                code_to_char_dict[ALPHABET_LENGTH] = c

        c_list.append(LINE_END_TOKEN)

    ALPHABET_LENGTH+=1 #dicts are zero-indexed, but we want to create vectors of size one greater

    return c_list, char_to_code_dict, code_to_char_dict, ALPHABET_LENGTH

def main():
    c_list, char_to_code_dict, code_to_char_dict, ALPHABET_LENGTH = readFile(DATAFILE)
    print char_to_code_dict

    c_list = c_list[:CHAR_LIMIT]
    c_list.append(LINE_END_TOKEN)

    minibatches = []
    sentences = []

    i=0
    while i<len(c_list)-SEQUENCE_LENGTH+1:
        sentences.append(c_list[i:i+SEQUENCE_LENGTH+1])
        i+=1

    i=0
    while i+MINIBATCH_SIZE<len(sentences):
        minibatches.append( sentences[i:i+MINIBATCH_SIZE] )
        i+=MINIBATCH_SIZE


    X_train = [np.dstack([np.asarray([one_hot(char_to_code_dict[c], ALPHABET_LENGTH) for c in sent[:-1]]) for sent in minibatch]) for minibatch in minibatches]
    y_train = [np.dstack([np.asarray([one_hot(char_to_code_dict[c], ALPHABET_LENGTH) for c in sent[1:]]) for sent in minibatch]) for minibatch in minibatches]


    print "\nDims of one minibatch: {0}".format(X_train[0].shape)

    push_vec = one_hot(char_to_code_dict[PUSH_CHAR], ALPHABET_LENGTH).reshape((ALPHABET_LENGTH,1))
    pop_vec = one_hot(char_to_code_dict[POP_CHAR], ALPHABET_LENGTH).reshape((ALPHABET_LENGTH,1))

    t1 = time.time()
    model = NWLSTM_Net(word_dim=ALPHABET_LENGTH, hidden_dim=HIDDEN_DIM, minibatch_dim=MINIBATCH_SIZE, bptt_truncate=BPTT_TRUNCATE,
                       num_layers=NUM_LAYERS, optimization=OPTIMIZATION, activation=ACTIVATION, want_stack=WANT_STACK,
                       stack_height=STACK_HEIGHT, push_vec=push_vec, pop_vec=pop_vec, dropout=DROPOUT)
    t2 = time.time()
    print "Compiling model took: {0:.0f} seconds\n".format(t2 - t1)


    t1 = time.time()
    model.train_model(X_train[0], y_train[0], LEARNING_RATE)
    t2 = time.time()
    print "One SGD step took: {0:.2f} milliseconds".format((t2 - t1) * 1000.)

    model.char_to_code_dict = char_to_code_dict
    model.code_to_char_dict = code_to_char_dict

    if MODEL_FILE != None:
        model = load_model_parameters_lstm(MODEL_FILE, minibatch_dim=MINIBATCH_SIZE)

    train_with_sgd(model, X_train, y_train, nepoch=NEPOCH, learning_rate=LEARNING_RATE)


if __name__=="__main__":
    main()