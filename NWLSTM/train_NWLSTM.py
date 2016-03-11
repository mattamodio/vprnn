#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, csv, itertools, operator, os, time, ast, datetime
import numpy as np
import theano
import theano.tensor as T
from utils import *
from NWLSTM_Net import NWLSTM_Net

# Data parameters
MAX_MINIBATCHES = False
MINIBATCH_SIZE = 2#128
SEQUENCE_LENGTH = 50#100
BPTT_TRUNCATE = -1#25

# Stack parameters
WANT_STACK = False
STACK_HEIGHT = 15
PUSH_CHAR = 'b'#u'\t'
POP_CHAR = 'c'#u'\n'
CONTEXT_TO_PUSH = 5
NULL = 'NULL_TOKEN'

# Layer parameters
HIDDEN_DIM = 256
NUM_LAYERS = 2
ACTIVATION = 'tanh'

# Optimization parameters
OPTIMIZATION = 'RMSprop' # RMSprop or SGD
LEARNING_RATE = .001
DROPOUT = .5
NEPOCH = 1000
EVAL_LOSS_AFTER = 1

# Data source parameters
#DATAFILE = '../data/war_and_peace.txt'
DATAFILE = '../data/tmp.txt'
MODEL_FILE = None


# theano.config.optimizer = 'None'
# theano.config.compute_test_value = 'raise'
# theano.config.exception_verbosity = 'high'

theano.config.warn_float64 = 'warn'
theano.config.floatX = 'float32'

print "Training NWLSTM with parameters:\
    \nMax Number of Minibatches: {0}\
    \nHidden Layer: {1}\
    \nNumber of Layers: {2}\
    \nStack: {8}\
    \nLearning Rate: {3}\
    \nDropout: {9}\
    \nOptimization: {11}\
    \nActivation: {12}\
    \nMinibatch size: {4}\
    \nSequence Length: {5}\
    \nTruncate gradient: {10}\
    \nNumber of epochs: {6}\
    \nLoading from model: {7}\n".format(MAX_MINIBATCHES, HIDDEN_DIM, NUM_LAYERS, LEARNING_RATE, MINIBATCH_SIZE,
        SEQUENCE_LENGTH, NEPOCH, MODEL_FILE, WANT_STACK, DROPOUT, BPTT_TRUNCATE, OPTIMIZATION, ACTIVATION)

def one_hot(x, dimensions):
    tmp = np.zeros(dimensions)

    tmp[x] = 1

    return tmp.astype('float32')

def nullOutMinibatch(minibatch, desired_length, desired_size):
    while len(minibatch)<desired_size:
        null_sequence = [NULL]*desired_length
        minibatch.append(null_sequence)
    return minibatch

def parseFileForCharacterDicts(filename):
    with open(filename, 'rb') as f:
        unique_chars = set()
        while True:
            chunk = f.read(256)
            for c in chunk:
                unique_chars.add(c)
            if not chunk: break

    code_to_char_dict = dict([(i,c) for i,c in enumerate(sorted(unique_chars))])
    char_to_code_dict = dict([(code_to_char_dict[k],k) for k in code_to_char_dict])

    ALPHABET_LENGTH = len(code_to_char_dict.keys()) + 1

    char_to_code_dict[NULL] = ALPHABET_LENGTH-1
    code_to_char_dict[ALPHABET_LENGTH-1] = NULL

    print char_to_code_dict

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
                    yield current_minibatch
                   
                    minibatches_yielded+=1
                    if MAX_MINIBATCHES and minibatches_yielded>=MAX_MINIBATCHES: break
                    
                    current_minibatch = []
                    #if minibatches_yielded%2000==0: print minibatches_yielded
                    continue

def readFileWithUnmatchedLefts(filename, dictsFile=None, outputDictsToFile=None):
    with open(filename, 'rb') as f:
        
        
        minibatches_yielded = 0
        current_minibatch = []
        current_sequence = [NULL]*(SEQUENCE_LENGTH-1)

        unmatched_pushes = []
        popped_buffers = []
        while True:
            c = f.read(1)
            if not c:
                break

            if c==PUSH_CHAR:
                current_sequence.append(c)
                push_buffer = current_sequence[-CONTEXT_TO_PUSH:]
                push_buffer.append(NULL)
                push_buffer = [char if char!=POP_CHAR else NULL for char in push_buffer]
                push_buffer = [SEQUENCE_LENGTH+1,push_buffer]
                unmatched_pushes.append( push_buffer )
                #print "\nPUSH BUFFER: {0}\n".format(push_buffer)
            elif c==POP_CHAR:
                pop_buffer = unmatched_pushes.pop()
                pop_buffer[0] = SEQUENCE_LENGTH+1
                popped_buffers.append(pop_buffer)
                #print "\nPOP BUFFER: {0}\n".format(pop_buffer)
                current_sequence.append(c)
            else:
                current_sequence.append(c)

            unmatched_pushes = [[max(0,counter-1), pushed] for counter,pushed in unmatched_pushes]
            popped_buffers = [[max(0,counter-1), pushed] for counter,pushed in popped_buffers]
            popped_buffers = filter(lambda x: x[0]!=0, popped_buffers)


            if len(current_sequence)>=SEQUENCE_LENGTH:
                prepended_mb = []
                    # for stack_item in unmatched_pushes:
                    #     if stack_item[0]==0:
                    #         prepended_mb.extend(stack_item[1])
                for stack_item in popped_buffers:
                    if stack_item[0]!=0:
                        prepended_mb.extend(stack_item[1])
                prepended_mb.extend(current_sequence)
                #print "".join(prepended_mb).replace("\t","<").replace("\r"," ").replace("\n",">")
                
                # if minibatch up to this point has a different dimension, null out and then yield it,
                # and start a new one for this sequence
                if len(current_minibatch)>0 and not len(current_minibatch[0])==len(prepended_mb):
                    current_minibatch = nullOutMinibatch(current_minibatch, len(current_minibatch[0]), MINIBATCH_SIZE)
                    #print ["".join(p).replace("\t","<").replace("\r"," ").replace("\n",">") for p in current_minibatch], [len(p) for p in current_minibatch]# yield it
                    yield current_minibatch
                    current_minibatch = []

                current_minibatch.append(prepended_mb)
                #current_minibatch.append(current_sequence) #add this sequence to current minibatch
                current_sequence = current_sequence[1:]
                
                if len(current_minibatch)>=MINIBATCH_SIZE: #next sequence starts a new minibatch
                    #print ["".join(p).replace("\t","<").replace("\r"," ").replace("\n",">") for p in current_minibatch], [len(p) for p in current_minibatch]
                    yield current_minibatch
                   
                    minibatches_yielded+=1
                    if MAX_MINIBATCHES and minibatches_yielded>=MAX_MINIBATCHES:
                        break
                    
                    current_minibatch = []
                    continue

def readFileGenerator(datafile, want_stack):
    if not want_stack:
        return readFile(datafile)
    else:
        return readFileWithUnmatchedLefts(datafile)

def main():
    print "Parsing for set of characters: {0}".format(DATAFILE)
    char_to_code_dict, code_to_char_dict, ALPHABET_LENGTH = parseFileForCharacterDicts(DATAFILE)
    print "Found {0} characters: {1}\n".format(ALPHABET_LENGTH, char_to_code_dict.keys())


    # for minibatch in readFileWithUnmatchedLefts(DATAFILE):
    #     for seq in minibatch:
    #         print len(seq), seq
    #     print 
    # sys.exit()

    print "Compiling model..."
    if MODEL_FILE != None:
        model = load_model_parameters_lstm(MODEL_FILE, minibatch_dim=MINIBATCH_SIZE)
    else:
        push_vec = one_hot(char_to_code_dict[PUSH_CHAR], ALPHABET_LENGTH).reshape((ALPHABET_LENGTH,1))
        pop_vec = one_hot(char_to_code_dict[POP_CHAR], ALPHABET_LENGTH).reshape((ALPHABET_LENGTH,1))
        null_vec = one_hot(char_to_code_dict[NULL], ALPHABET_LENGTH).reshape((ALPHABET_LENGTH,1))
        t1 = time.time()
        model = NWLSTM_Net(word_dim=ALPHABET_LENGTH, hidden_dim=HIDDEN_DIM, minibatch_dim=MINIBATCH_SIZE, bptt_truncate=BPTT_TRUNCATE,
                           num_layers=NUM_LAYERS, optimization=OPTIMIZATION, activation=ACTIVATION, want_stack=WANT_STACK,
                           stack_height=STACK_HEIGHT, push_vec=push_vec, pop_vec=pop_vec, null_vec=null_vec, dropout=DROPOUT)
        t2 = time.time()
        model.char_to_code_dict = char_to_code_dict
        model.code_to_char_dict = code_to_char_dict
        print "Finished! Compiling model took: {0:.0f} seconds\n".format(t2 - t1)


    losses=[]
    counter = 0
    for epoch in xrange(NEPOCH):
        #print "Epoch: {0}".format(epoch)
        for minibatch in readFileGenerator(DATAFILE, WANT_STACK):
            #print "Current minibatch: {0}".format([[str(c) for c in seq] for seq in minibatch])

            x = np.dstack([np.asarray([one_hot(char_to_code_dict[c], ALPHABET_LENGTH) for c in seq[:-1]]) for seq in minibatch])
            y = np.dstack([np.asarray([one_hot(char_to_code_dict[c], ALPHABET_LENGTH) for c in seq[1:]]) for seq in minibatch])


            if counter%EVAL_LOSS_AFTER==0:
                # print minibatch
                if epoch==0: print "Dims of one minibatch: {0}".format(x.shape)
                t1 = time.time()
                model.train_model(x, y, LEARNING_RATE)
                t2 = time.time()
                if counter%(EVAL_LOSS_AFTER*10)==0: print "One SGD step took: {0:.2f} milliseconds".format((t2 - t1) * 1000.)

                # probs = model.forward_propagation(x)
                # print probs
                loss, l1_loss, l2_loss = model.loss_for_minibatch(x,y)
                losses.append((epoch, loss))
                dt = datetime.datetime.now().strftime("%m-%d___%H-%M-%S")
                print "{0}: Loss on first minibatch after epoch={1}: {2:.1f},{3:.1f},{4:.1f}".format(dt, epoch, loss, l1_loss, l2_loss)
                
                save_model_parameters_lstm("saved_model_parameters/NWLSTM_savedparameters_{0:.1f}__{1}.npz".format(loss, dt), model)
            else:
                model.train_model(x, y, LEARNING_RATE)

            counter+=1


    print "{0}: Loss after all epochs: {1:.1f}".format(dt, loss)
    save_model_parameters_lstm("saved_model_parameters/NWLSTM_savedparameters_{0:.1f}__{1}.npz".format(loss, dt), model)          



if __name__=="__main__":
    main()