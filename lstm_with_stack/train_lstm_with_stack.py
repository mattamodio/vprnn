#! /usr/bin/env python
import sys
import csv
import itertools
import operator
import numpy as np
import nltk
import os
import time
from datetime import datetime
from utils_lstm_with_stack import *
from lstm_with_stack import LSTM_with_stack
from generate_from_lstm_with_stack import generate_sentence

#_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '500'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.001'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))

_MODEL_FILE = os.environ.get('MODEL_FILE', )
_MODEL_FILE = os.environ.get('MODEL_FILE', 'saved_model_parameters/lstm-_with_stack-theano-500-82-2016-02-15-14-11-47.npz')

def one_hot(x, dimensions):
    tmp = np.zeros(dimensions)
    tmp[x] = 1
    return tmp

def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=1, evaluate_loss_after=1, num_sentences=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            # if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
            #     learning_rate = learning_rate * 0.5  
            #     print "Setting learning rate to %f" % learning_rate
            # ADDED! Saving model oarameters
            save_model_parameters_lstm("saved_model_parameters/lstm-_with_stack-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
            # for i in range(num_sentences):
            #     sent = []
            #     sent = generate_sentence(model, char_to_code_dict, code_to_char_dict, line_start_token, line_end_token, ALPHABET_LENGTH+1, sample_limit=50)
            #     print "".join(sent)
            # sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

line_start_token = "LINE_START"
line_end_token = "LINE_END"
ALPHABET_LENGTH = 1

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
with open('../data/mlb.xml', 'rb') as f:
    char_to_code_dict = {line_start_token: 0, line_end_token: 1}
    code_to_char_dict = {0: line_start_token, 1: line_end_token}
    # sentences = []
    # for trainingLine in f:
    #     c_list = [line_start_token]
    #     for c in trainingLine:
    #         c_list.append(c)
    #         if c not in char_to_code_dict:
    #             ALPHABET_LENGTH+=1
    #             char_to_code_dict[c] = ALPHABET_LENGTH
    #             code_to_char_dict[ALPHABET_LENGTH] = c
    #     c_list.append(line_end_token)
    #     sentences.append(c_list)


    filestring = f.read()
    c_list = [line_start_token]
    for c in filestring:
        c_list.append(c)
        if c not in char_to_code_dict:
            ALPHABET_LENGTH+=1
            char_to_code_dict[c] = ALPHABET_LENGTH
            code_to_char_dict[ALPHABET_LENGTH] = c
    c_list.append(line_end_token)
    c_list = c_list

#print "Parsed %d lists of characters." % (len(c_list))
print "Found {0} characters: {1}".format(ALPHABET_LENGTH, char_to_code_dict.keys())

dictFile = 'dictFile.txt'
with open(dictFile, 'w+') as f:
    f.write(str(char_to_code_dict))
    f.write("\n")
    f.write(str(code_to_char_dict))
print "Wrote character-to-code dicts to {0}".format(dictFile)


c_list = c_list[:1000]
c_list.append(line_end_token)
# Create the training data
minibatch_size = 1
sequence_length = 50
minibatches = []
sentences = []

i=0
while i<len(c_list)-sequence_length+1:
    sentences.append(c_list[i:i+sequence_length+1])
    i+=1
print "Created sequences."

i=0
while i+minibatch_size<len(sentences):
    minibatches.append( sentences[i:i+minibatch_size] )
    i+=minibatch_size
print "Created minibatches."

X_train = [np.dstack([np.asarray([one_hot(char_to_code_dict[c], ALPHABET_LENGTH+1) for c in sent[:-1]]) for sent in minibatch]) for minibatch in minibatches]
y_train = [np.dstack([np.asarray([one_hot(char_to_code_dict[c], ALPHABET_LENGTH+1) for c in sent[1:]]) for sent in minibatch]) for minibatch in minibatches]
print "Created training set."

X_train = [minibatch.astype('int32') for minibatch in X_train]
y_train = [minibatch.astype('int32') for minibatch in y_train]
print "Changed dtype of training set."

print "Dims of one minibatch: {0}".format(X_train[0].shape)

push_vec = one_hot(char_to_code_dict['<'], ALPHABET_LENGTH+1)
pop_vec = one_hot(char_to_code_dict['>'], ALPHABET_LENGTH+1)

model = LSTM_with_stack(ALPHABET_LENGTH+1, hidden_dim=_HIDDEN_DIM, minibatch_size=minibatch_size, bptt_truncate=sequence_length, push_vec=push_vec, pop_vec=pop_vec)
#model = lstm_theano.LSTMTheano(ALPHABET_LENGTH+1, hidden_dim=_HIDDEN_DIM)
# t1 = time.time()
# model.sgd_step(X_train[0], y_train[0], _LEARNING_RATE)
# t2 = time.time()
# print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

if _MODEL_FILE != None:
    model = load_model_parameters_lstm(_MODEL_FILE, minibatch_size=minibatch_size)

train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)