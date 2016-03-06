#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, csv, itertools, operator, os, time, ast, datetime
import numpy as np
import theano
import theano.tensor as T
from utils import *
from NWLSTM_Net import NWLSTM_Net

# Data parameters
MAX_MINIBATCHES = 10
MINIBATCH_SIZE = 100
SEQUENCE_LENGTH = 100
BPTT_TRUNCATE = -1

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
DROPOUT = .25
NEPOCH = 1000
EVAL_LOSS_AFTER = 1

# Data source parameters
DATAFILE = '../data/war_and_peace.txt'
MODEL_FILE = None


# theano.config.optimizer = 'None'
# theano.config.compute_test_value = 'raise'
# theano.config.exception_verbosity = 'high'
# # #theano.config.profile = True
theano.config.warn_float64 = 'warn'
theano.config.floatX = 'float32'

print "Training NWLSTM with parameters:\
    \nMax Number of Minibatches: {0}\
    \nHidden Layer: {1}\
    \nNumber of Layers: {2}\
    \nStack: {8}\
    \nLearning Rate: {3}\
    \nDropout: {9}\
    \nMinibatch size: {4}\
    \nSequence Length: {5}\
    \nTruncate gradient: {10}\
    \nNumber of epochs: {6}\
    \nLoading from model: {7}\n".format(MAX_MINIBATCHES, HIDDEN_DIM, NUM_LAYERS, LEARNING_RATE, MINIBATCH_SIZE,
        SEQUENCE_LENGTH, NEPOCH, MODEL_FILE, WANT_STACK, DROPOUT, BPTT_TRUNCATE)

def one_hot(x, dimensions):
    tmp = np.zeros(dimensions)
    tmp[x] = 1
    return tmp.astype('float32')

# def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=1, evaluate_loss_after=1000):
    # # We keep track of the losses so we can plot them later
    # losses = []
    # examples_per_epoch = (CHAR_LIMIT - SEQUENCE_LENGTH) / MINIBATCH_SIZE
    # percent_progress_points = set([int((i/10.) * examples_per_epoch) for i in range(1,10)])
    # print "Will print progress at steps: {0}".format(sorted(list(percent_progress_points)))

    # for epoch in range(nepoch):
    #     num_examples_seen = 0
    #     percent_progress = 1
        
    #     if (epoch % evaluate_loss_after == 0): # Optionally evaluate the loss
    #         loss = model.calculate_loss(X_train, y_train)
    #         losses.append((num_examples_seen, loss))
    #         time = datetime.datetime.now().strftime("%m-%d___%H-%M-%S")
    #         print "{0}: Loss after epoch={1}: {2}".format(time, epoch, loss)
    #         save_model_parameters_lstm("saved_model_parameters/NWLSTM_savedparameters_{0:.1f}__{1}.npz".format(loss, time), model)

    #     for i in xrange(len(y_train)): # One SGD step
    #         model.train_model(X_train[i], y_train[i], learning_rate)
    #         num_examples_seen += 1
    #         # if num_examples_seen in percent_progress_points:
    #         #     print "{0}% finished".format(str(percent_progress)+"0")
    #         #     percent_progress+=1
    #         #     sys.stdout.flush()


def parseFileForCharacterDicts(filename):
    with open(filename, 'rb') as f:
        char_to_code_dict = {}
        code_to_char_dict = {}
        ALPHABET_LENGTH = 0
        while True:
            chunk = f.read(256)
            for c in chunk:
                if c not in char_to_code_dict:
                    char_to_code_dict[c] = ALPHABET_LENGTH
                    code_to_char_dict[ALPHABET_LENGTH] = c
                    ALPHABET_LENGTH+=1
            if not chunk: break

    #ALPHABET_LENGTH+=1 #dicts are zero-indexed, but we want to create vectors of size one greater

    return char_to_code_dict, code_to_char_dict, ALPHABET_LENGTH

def readFile(filename, dictsFile=None, outputDictsToFile=None):
    with open(filename, 'rb') as f:
        
        minibatches_yielded = 0
        current_minibatch = []
        current_sequence = []
        while True:
            c = f.read(1)
            if not c: break


            current_sequence.append(c)

            if len(current_sequence)>=SEQUENCE_LENGTH:
                current_minibatch.append(current_sequence) #add this sequence to current minibatch
                current_sequence = current_sequence[1:]
                
                if len(current_minibatch)>=MINIBATCH_SIZE: #next sequence starts a new minibatch
                    #print [[str(c) for c in seq] for seq in current_minibatch]
                    yield current_minibatch
                   
                    minibatches_yielded+=1
                    if minibatches_yielded>=MAX_MINIBATCHES: break
                    
                    current_minibatch = []
                    if minibatches_yielded%100==0: print minibatches_yielded
                    continue


                

def readFileWithUnmatchedLefts(filename, dictsFile=None, outputDictsToFile=None):
    with open(filename, 'rb') as f:
        
        minibatches_yielded = 0
        current_minibatch = []
        current_sequence = []

        unmatched_pushes = []
        pop_buffer = []
        while True:
            c = f.read(1)
            if not c: break

            if c==PUSH_CHAR:
                push_buffer = current_sequence
                push_buffer.append('NULL')
                unmatched_pushes.append(push_buffer)
                print "PUSH BUFFER: {0}".format(push_buffer)
            elif c==POP_CHAR:
                pop_buffer = unmatched_pushes.pop()
                print "POP BUFFER: {0}".format(pop_buffer)
            else:
                print c
            current_sequence.append(c)

            if len(current_sequence)>=SEQUENCE_LENGTH:


                current_sequence_with_prepended_buffer = pop_buffer[:]
                current_sequence_with_prepended_buffer.extend(current_sequence)
                current_minibatch.append(current_sequence_with_prepended_buffer)



                #current_minibatch.append(current_sequence) #add this sequence to current minibatch
                current_sequence = current_sequence[1:]
                
                if len(current_minibatch)>=MINIBATCH_SIZE: #next sequence starts a new minibatch
                    #print [[str(c) for c in seq] for seq in current_minibatch]


                    #yield current_minibatch
                    print current_minibatch
                   
                    minibatches_yielded+=1
                    if minibatches_yielded>=MAX_MINIBATCHES: break
                    
                    current_minibatch = []
                    continue

        return None,None,None


def main():
    print "Parsing for set of characters: {0}".format(DATAFILE)
    char_to_code_dict, code_to_char_dict, ALPHABET_LENGTH = parseFileForCharacterDicts(DATAFILE)
    print "Found {0} characters: {1}\n".format(len(char_to_code_dict), char_to_code_dict.keys())


    #char_to_code_dict, code_to_char_dict, ALPHABET_LENGTH = readFileWithUnmatchedLefts(DATAFILE)
    #sys.exit()


    print "Compiling model..."
    if MODEL_FILE != None:
        model = load_model_parameters_lstm(MODEL_FILE, minibatch_dim=MINIBATCH_SIZE)
    else:
        push_vec = one_hot(char_to_code_dict[PUSH_CHAR], ALPHABET_LENGTH).reshape((ALPHABET_LENGTH,1))
        pop_vec = one_hot(char_to_code_dict[POP_CHAR], ALPHABET_LENGTH).reshape((ALPHABET_LENGTH,1))
        t1 = time.time()
        model = NWLSTM_Net(word_dim=ALPHABET_LENGTH, hidden_dim=HIDDEN_DIM, minibatch_dim=MINIBATCH_SIZE, bptt_truncate=BPTT_TRUNCATE,
                           num_layers=NUM_LAYERS, optimization=OPTIMIZATION, activation=ACTIVATION, want_stack=WANT_STACK,
                           stack_height=STACK_HEIGHT, push_vec=push_vec, pop_vec=pop_vec, dropout=DROPOUT)
        t2 = time.time()
        model.char_to_code_dict = char_to_code_dict
        model.code_to_char_dict = code_to_char_dict
        print "Finished! Compiling model took: {0:.0f} seconds\n".format(t2 - t1)


    losses=[]
    for epoch in xrange(NEPOCH):
        #print "Epoch: {0}".format(epoch)
        first = True
        for minibatch in readFile(DATAFILE):
            #print "Current minibatch: {0}".format([[str(c) for c in seq] for seq in minibatch])

            x = np.dstack([np.asarray([one_hot(char_to_code_dict[c], ALPHABET_LENGTH) for c in seq[:-1]]) for seq in minibatch])
            y = np.dstack([np.asarray([one_hot(char_to_code_dict[c], ALPHABET_LENGTH) for c in seq[1:]]) for seq in minibatch])


            if first and epoch%EVAL_LOSS_AFTER==0:
                if epoch==0: print "Dims of one minibatch: {0}".format(x.shape)
                t1 = time.time()
                model.train_model(x, y, LEARNING_RATE)
                t2 = time.time()
                if epoch==0: print "One SGD step took: {0:.2f} milliseconds\n".format((t2 - t1) * 1000.)


                loss = model.loss_for_minibatch(x,y)
                losses.append((epoch, loss))
                dt = datetime.datetime.now().strftime("%m-%d___%H-%M-%S")
                print "{0}: Loss on first minibatch after epoch={1}: {2:.1f}".format(dt, epoch, loss)
                save_model_parameters_lstm("saved_model_parameters/NWLSTM_savedparameters_{0:.1f}__{1}.npz".format(loss, dt), model)
                first = False
            else:
                model.train_model(x, y, LEARNING_RATE)


    print "{0}: Loss after all epochs: {1:.1f}".format(dt, loss)
    save_model_parameters_lstm("saved_model_parameters/NWLSTM_savedparameters_{0:.1f}__{1}.npz".format(loss, dt), model)          



if __name__=="__main__":
    main()