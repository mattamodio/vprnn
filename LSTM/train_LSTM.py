#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, csv, itertools, operator, os, time, ast, datetime
import numpy as np
import theano
import theano.tensor as T
from utils import *
from LSTM_Net import LSTM_Net
from generate_from_LSTM import generate_sentence

# Data parameters
MINIBATCH_SIZE = 100
SEQUENCE_LENGTH = 100 #SEQUENCE_LENGTH>=MINIBATCH_SIZE REQUIRED
BPTT_TRUNCATE = 50

# Layer parameters
HIDDEN_DIM = 512
NUM_LAYERS = 2
ACTIVATION = 'tanh'

# Optimization parameters
OPTIMIZATION = 'RMSprop' # RMSprop or SGD
LEARNING_RATE = .01
DROPOUT = .2
MAX_MINIBATCHES = 10000
EVAL_LOSS_AFTER = 100
L1_REGULARIZATION = 0.#0001
L2_REGULARIZATION = 0.001


# Data source parameters
# DATAFILE = '../data/tmp.txt'
# LOSSFILE = '../data/tmp_loss.txt'
DATAFILE = '../data/simcity.txt'
LOSSFILE = '../data/simcity_loss.txt'
# DATAFILE = '../data/tex.txt'
# LOSSFILE = '../data/tex_loss.txt'
# DATAFILE = '../data/war_and_peace.txt'
# LOSSFILE = '../data/war_and_peace_loss.txt'
# DATAFILE = '../data/mlb.xml'
# LOSSFILE = '../data/mlb_loss.txt'
MODEL_FILE = None #'saved_model_parameters/LSTM_savedparameters_229707.4__04-04___12-34-50.npz'

# Sampling parameters
SAMPLE = True
STARTING_STRING = open(LOSSFILE).read()

SAMPLE_EVERY = 100
SAMPLE_LIMIT = 500
SAMPLE_NUMBER = 1
SOFTMAX_TEMPS = [.5]

# theano.config.optimizer = 'None'
# theano.config.compute_test_value = 'raise'
# theano.config.exception_verbosity = 'high'

# theano.config.profile='True'

theano.config.warn_float64 = 'warn'
theano.config.floatX = 'float32'

np.random.seed = 1

print "Training LSTM with parameters:\
    \nMax Number of Minibatches: {0}\
    \nMinibatch size: {1}\
    \nSequence Length: {2}\
    \nTruncate gradient: {3}\
    \nHidden Layer: {4}\
    \nNumber of Layers: {5}\
    \nActivation: {6}\
    \nOptimization: {7}\
    \nLearning Rate: {8}\
    \nDropout: {9}\
    \nEvaluate loss every: {10}\
    \nL1 Regularization: {11}\
    \nL2 Regularization: {12}\
    \nLoading from model: {13}\
    \nDatafile: {14}\
    \nSample: {15}, every {16} iterations\n".format(MAX_MINIBATCHES, MINIBATCH_SIZE, SEQUENCE_LENGTH, BPTT_TRUNCATE, HIDDEN_DIM, NUM_LAYERS,
        ACTIVATION, OPTIMIZATION, LEARNING_RATE, DROPOUT, EVAL_LOSS_AFTER, L1_REGULARIZATION, L2_REGULARIZATION, MODEL_FILE, DATAFILE, SAMPLE, SAMPLE_EVERY)

def one_hot(x, dimensions):
    tmp = np.zeros(dimensions)
    tmp[x] = 1
    return tmp.astype('float32')

def getCharDicts():
    NULL = 'NULL'
    chars = [NULL, ' ', '\t', '\r', '\n'] + [chr(x) for x in xrange(33,127)]
    code_to_char_dict = dict([(i,c) for i,c in enumerate(sorted(chars))])
    char_to_code_dict = dict([(code_to_char_dict[k],k) for k in code_to_char_dict])

    ALPHABET_LENGTH = len(code_to_char_dict.keys())

    return char_to_code_dict, code_to_char_dict, ALPHABET_LENGTH

def getCode(char, char_to_code_dict):
    if char in char_to_code_dict:
        code = char_to_code_dict[char]
    else:
        code = char_to_code_dict['NULL']
    return code

def readFile(filename, char_to_code_dict, minibatch_size):

    with open(filename) as f:
        filestring = f.read()

    np_data = np.zeros((len(filestring), len(char_to_code_dict))).astype('float32')

    c_index=0
    for c in filestring:
        code = getCode(c, char_to_code_dict)
        np_data[c_index, code] = 1
        c_index+=1


    minibatches_yielded = 0
    startend_positions = [(start,start+SEQUENCE_LENGTH) for start in xrange(minibatch_size)]

    end_file = False
    while True:
        x_s = []
        y_s = []
        for start,end in startend_positions:
            if (end+1)>=np_data.shape[0]:
                end_file=True
                break

            x = np_data[start:end,:]
            y = np_data[start+1:end+1,:]
            x_s.append(x)
            y_s.append(y)

        if end_file: break

        x_s = np.stack(x_s, axis=2)
        y_s = np.stack(y_s, axis=2)

        yield x_s,y_s

        minibatches_yielded+=1
        if MAX_MINIBATCHES and minibatches_yielded>=MAX_MINIBATCHES: break

        startend_positions = [(x+minibatch_size,y+minibatch_size) for x,y in startend_positions]

def calculateLoss(filename, model, counter):
    with open(filename) as f:
        seq = f.read()
        minibatch = [seq]*int(model.minibatch_dim)

    loss_x = np.dstack([np.asarray([one_hot(model.getCode(c), model.word_dim) for c in seq[:-1]]) for seq in minibatch])
    loss_y = np.dstack([np.asarray([one_hot(model.getCode(c), model.word_dim) for c in seq[1:]]) for seq in minibatch])

    softmax_temp = 1
    h_prev = np.zeros((model.num_layers,model.hidden_dim,model.minibatch_dim)).astype('float32')
    c_prev = np.zeros((model.num_layers,model.hidden_dim,model.minibatch_dim)).astype('float32')

    loss, l1_loss, l2_loss = model.loss_for_minibatch(loss_x, loss_y, h_prev, c_prev, softmax_temp)

    loss_per_char = loss / (int(model.minibatch_dim) * len(seq))

    dt = datetime.datetime.now().strftime("%m-%d___%H-%M-%S")
    print "{0}: Loss after examples={1}:  {2:.0f}    {3:.0f}    {4:.0f}    {5:.5f}".format(dt, counter, loss, l1_loss, l2_loss, loss_per_char)
    save_model_parameters_lstm("saved_model_parameters/LSTM_savedparameters_{0:.1f}__{1}.npz".format(loss, dt), model)

def pretrain(filename, model):
    model.build_pretrain()

    with open(filename) as f:
        filestring = f.read()

    counter=1
    for c in filestring[:1000]:
        code = getCode(c, model.char_to_code_dict)
        x = np.zeros((1, model.word_dim, model.minibatch_dim)).astype('float32')
        x[:,code,:] = 1

        pretrain_o, pretrain_h_init, pretrain_c_init = model.pretrain_model(x, .01, 1)

        if counter%100==0:
            print code
            #print pretrain_o
            print np.argmax(pretrain_o, axis=0)

        counter+=1

    return pretrain_h_init, pretrain_c_init


def main():
    char_to_code_dict, code_to_char_dict, ALPHABET_LENGTH = getCharDicts()
    print "char_to_code_dict: {0}\ncode_to_char_dict: {1}\n".format(char_to_code_dict, code_to_char_dict)

    print "Compiling model..."
    t1 = time.time()
    if MODEL_FILE != None:
        model = load_model_parameters_lstm(MODEL_FILE)
    else:
        model = LSTM_Net(word_dim=ALPHABET_LENGTH, hidden_dim=HIDDEN_DIM, minibatch_dim=MINIBATCH_SIZE, bptt_truncate=BPTT_TRUNCATE,
                           num_layers=NUM_LAYERS, optimization=OPTIMIZATION, activation=ACTIVATION, dropout=DROPOUT,
                           l1_rate=L1_REGULARIZATION, l2_rate=L2_REGULARIZATION)
        model.char_to_code_dict = char_to_code_dict
        model.code_to_char_dict = code_to_char_dict
    t2 = time.time()
    print "Finished! Compiling model took: {0:.0f} seconds\n".format(t2 - t1)

    # model.build_pretrain()
    # counter=1
    # for x,y in readFile(DATAFILE, char_to_code_dict):
    #     pretrain_o, pretrain_h_init, pretrain_c_init = model.pretrain_model(x, .1, 1)

    #     if counter%1000==0:
    #         break
    #         # print np.argmax(x, axis=1)
    #         # print np.argmax(pretrain_o, axis=1)

    #     counter+=1
    # #sys.exit()
    # #h_init_pretrain, c_init_pretrain = pretrain(DATAFILE, model)

    losses = []
    counter = 0
    softmax_temp = 1
    while counter<MAX_MINIBATCHES:
        h_prev = np.zeros((model.num_layers,model.hidden_dim,model.minibatch_dim)).astype('float32')
        c_prev = np.zeros((model.num_layers,model.hidden_dim,model.minibatch_dim)).astype('float32')
        # h_prev = pretrain_h_init
        # c_prev = pretrain_c_init

        for x,y in readFile(DATAFILE, char_to_code_dict, model.minibatch_dim):

            if counter%EVAL_LOSS_AFTER==0:
                if counter==0: print "Dims of one minibatch: {0}".format(x.shape)
                t1 = time.time()

                h_prev,c_prev = model.train_model(x, y, h_prev, c_prev, LEARNING_RATE, softmax_temp)

                t2 = time.time()
                if counter%(EVAL_LOSS_AFTER*10)==0: print "One SGD step took: {0:.2f} milliseconds".format((t2 - t1) * 1000.)


                calculateLoss(LOSSFILE, model, counter)
            else:
                h_prev,c_prev = model.train_model(x, y, h_prev, c_prev, LEARNING_RATE, softmax_temp)


            if SAMPLE and (counter!=0) and counter%SAMPLE_EVERY==0:
                for softmax_temp in SOFTMAX_TEMPS:
                    print "\nSampling sentence with softmax {0}".format(softmax_temp)
                    for _ in xrange(SAMPLE_NUMBER):
                        sent = generate_sentence(model, sample_limit=SAMPLE_LIMIT, softmax_temp=softmax_temp, starting_string=STARTING_STRING)
                        print "".join(sent)
                print

            counter+=1
            

    calculateLoss(LOSSFILE, model, counter)          



if __name__=="__main__":
    main()