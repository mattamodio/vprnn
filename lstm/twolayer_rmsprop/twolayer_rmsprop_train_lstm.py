#! /usr/bin/env python
import sys, csv, itertools, operator, os, time, ast
import numpy as np
from datetime import datetime
from twolayer_rmsprop_utils import *
from twolayer_rmsprop_lstm_theano import LSTMTheano

_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '500'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.001'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE', )
#_MODEL_FILE = os.environ.get('MODEL_FILE', 'saved_model_parameters/lstm_rmsprop-theano-500-83-2016-02-19-18-13-55.npz')

CHARS_TO_TRAIN = 100000
MINIBATCH_SIZE = 100
SEQUENCE_LENGTH = 100
LINE_START_TOKEN = "LINE_START"
LINE_END_TOKEN = "LINE_END"
# DATAFILE = '../../data/gatsby.txt'
# DICTFILE = None #'../dictFile.txt'
# OUTPUTDICTFILE = '../gatsbydictfile.txt'

DATAFILE = '../../data/mlb.xml'
DICTFILE = '../dictFile.txt'
OUTPUTDICTFILE = None


print "Training Two-layer Rmsprop LSTM with parameters:\
    \nLength of training set: {0}\
    \nHidden Layer: {1}\
    \nLearning Rate: {2}\
    \nMinibatch size: {3}\
    \nSequence Length: {4}\
    \nNumber of epochs: {5}\
    \nLoading from model: {6}".format(CHARS_TO_TRAIN, _HIDDEN_DIM, _LEARNING_RATE, MINIBATCH_SIZE, SEQUENCE_LENGTH, _NEPOCH, _MODEL_FILE)

def one_hot(x, dimensions):
    tmp = np.zeros(dimensions)
    tmp[x] = 1
    return tmp

def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=1, evaluate_loss_after=1, num_sentences=5):
    # We keep track of the losses so we can plot them later
    losses = []
    examples_per_epoch = (CHARS_TO_TRAIN - SEQUENCE_LENGTH) / MINIBATCH_SIZE
    decimal_points = [(i/10.) * examples_per_epoch for i in range(1,10)]
    for epoch in range(nepoch):
        num_examples_seen = 0
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
            # ADDED! Saving model parameters
            save_model_parameters_lstm("saved_model_parameters/lstm_rmsprop-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in xrange(len(y_train)):
            # One SGD step
            model.train_model(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
            if num_examples_seen in decimal_points:
                print num_examples_seen
                sys.stdout.flush()

def readFile(filename, dictsFile=None, outputDictsToFile=None):

    # Read data and make dicts for encodings if they are not provided
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    #print "Reading CSV file..."
    with open(filename, 'rb') as f:
        if not dictsFile:
            char_to_code_dict = {LINE_START_TOKEN: 0, LINE_END_TOKEN: 1}
            code_to_char_dict = {0: LINE_START_TOKEN, 1: LINE_END_TOKEN}
            ALPHABET_LENGTH = 1

        filestring = f.read()
        c_list = [LINE_START_TOKEN]
        for c in filestring:
            c_list.append(c)

            if not dictsFile:
                if c not in char_to_code_dict:
                    ALPHABET_LENGTH+=1
                    char_to_code_dict[c] = ALPHABET_LENGTH
                    code_to_char_dict[ALPHABET_LENGTH] = c

        c_list.append(LINE_END_TOKEN)

    if dictsFile:
        print "Loaded encoding dictionaries from: {0}".format(dictsFile)
        with open(dictsFile) as f:
            dicts = []
            for line in f:
                line = ast.literal_eval(line)
                dicts.append(line)
        char_to_code_dict, code_to_char_dict = dicts
        
    else:
        print "Created encoding dictionaries with {0} characters: {1}".format(ALPHABET_LENGTH, char_to_code_dict.keys())
        with open(outputDictsToFile, 'w+') as f:
            f.write(str(char_to_code_dict))
            f.write("\n")
            f.write(str(code_to_char_dict))
        print "Wrote character-to-code dicts to {0}".format(outputDictsToFile)

    ALPHABET_LENGTH = len(char_to_code_dict.keys())+1

    return c_list, char_to_code_dict, code_to_char_dict, ALPHABET_LENGTH


def main():
    c_list, char_to_code_dict, code_to_char_dict, ALPHABET_LENGTH = readFile(DATAFILE, dictsFile=DICTFILE, outputDictsToFile=OUTPUTDICTFILE)

    c_list = c_list[:CHARS_TO_TRAIN]
    c_list.append(LINE_END_TOKEN)
    # Create the training data
    minibatches = []
    sentences = []

    i=0
    while i<len(c_list)-SEQUENCE_LENGTH+1:
        sentences.append(c_list[i:i+SEQUENCE_LENGTH+1])
        i+=1
    # print "Created sequences."

    i=0
    while i+MINIBATCH_SIZE<len(sentences):
        minibatches.append( sentences[i:i+MINIBATCH_SIZE] )
        i+=MINIBATCH_SIZE
    # print "Created minibatches."

    X_train = [np.dstack([np.asarray([one_hot(char_to_code_dict[c], ALPHABET_LENGTH) for c in sent[:-1]]) for sent in minibatch]) for minibatch in minibatches]
    y_train = [np.dstack([np.asarray([one_hot(char_to_code_dict[c], ALPHABET_LENGTH) for c in sent[1:]]) for sent in minibatch]) for minibatch in minibatches]
    # print "Created training set."

    X_train = [minibatch.astype('int32') for minibatch in X_train]
    y_train = [minibatch.astype('int32') for minibatch in y_train]
    # print "Changed dtype of training set."

    print "\nDims of one minibatch: {0}".format(X_train[0].shape)

    model = LSTMTheano(ALPHABET_LENGTH, hidden_dim=_HIDDEN_DIM, minibatch_size=MINIBATCH_SIZE, bptt_truncate=min(SEQUENCE_LENGTH,50))

    t1 = time.time()
    model.train_model(X_train[0], y_train[0], _LEARNING_RATE)
    t2 = time.time()
    print "One SGD step took: %f milliseconds" % ((t2 - t1) * 1000.)

    if _MODEL_FILE != None:
        model = load_model_parameters_lstm(_MODEL_FILE, minibatch_size=MINIBATCH_SIZE)

    train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)


if __name__=="__main__":
    main()