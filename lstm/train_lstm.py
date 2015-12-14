#! /usr/bin/env python
import sys
sys.path.append('rnn-tutorial-rnnlm')
import csv
import itertools
import operator
import numpy as np
import nltk
import os
import time
from datetime import datetime
from utils import *
from lstm_theano import LSTMTheano

#_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '500'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.1'))
_NEPOCH = int(os.environ.get('NEPOCH', '1000'))
_MODEL_FILE = os.environ.get('MODEL_FILE')
#_MODEL_FILE = os.environ.get('MODEL_FILE', 'saved_model_parameters/lstm-theano-100-57-2015-12-13-01-53-45.npz')

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=10):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        print "Epoch {0}".format(epoch)
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = max(learning_rate * 0.5, .001)
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_lstm("saved_model_parameters/lstm-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
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
    sentences = []
    for trainingLine in f:
        c_list = [line_start_token]
        trainingLine = trainingLine.lower()
        for c in trainingLine:
            c_list.append(c)
            if c not in char_to_code_dict:
                ALPHABET_LENGTH+=1
                char_to_code_dict[c] = ALPHABET_LENGTH
                code_to_char_dict[ALPHABET_LENGTH] = c
        c_list.append(line_end_token)
        sentences.append(c_list)
    # filestring = f.read()
    # c_list = [line_start_token]
    # for c in filestring:
    #     c_list.append(c)
    #     if c not in char_to_code_dict:
    #         ALPHABET_LENGTH+=1
    #         char_to_code_dict[c] = ALPHABET_LENGTH
    #         code_to_char_dict[ALPHABET_LENGTH] = c
    # c_list.append(line_end_token)
    # sentences = [c_list]

print "Parsed %d lists of characters." % (len(sentences))
print "Found {0} characters: {1}".format(ALPHABET_LENGTH, char_to_code_dict.keys())

dictFile = 'dictFile.txt'
with open(dictFile, 'w+') as f:
    f.write(str(char_to_code_dict))
    f.write("\n")
    f.write(str(code_to_char_dict))
print "Wrote character-to-code dicts to {0}".format(dictFile)



# Create the training data
X_train = np.asarray([[char_to_code_dict[c] for c in sent[:-1]] for sent in sentences])#, dtype='int32')
y_train = np.asarray([[char_to_code_dict[c] for c in sent[1:]] for sent in sentences])#, dtype='int32')

print "Created training set."

model = LSTMTheano(ALPHABET_LENGTH+1, hidden_dim=_HIDDEN_DIM)
#model = lstm_theano.LSTMTheano(ALPHABET_LENGTH+1, hidden_dim=_HIDDEN_DIM)
t1 = time.time()
model.sgd_step(X_train[0], y_train[0], _LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

if _MODEL_FILE != None:
    model = load_model_parameters_lstm(_MODEL_FILE)

train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)