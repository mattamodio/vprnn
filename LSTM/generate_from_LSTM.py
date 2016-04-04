#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, csv, itertools, operator, os, time, ast
import numpy as np
from datetime import datetime
from utils import *
import theano


# theano.config.optimizer = 'None'
# theano.config.compute_test_value = 'raise'
# theano.config.exception_verbosity = 'high'
# # #theano.config.profile = True
# theano.config.warn_float64 = 'warn'
# theano.config.floatX = 'float32'


def generate_sentence(model, starting_string, sample_limit=50, sample_from_distribution=True, softmax_temp=1):

    char_to_code_dict, code_to_char_dict, word_dim, minibatch_dim = model.char_to_code_dict, model.code_to_char_dict, model.word_dim, model.minibatch_dim

    def one_hot(x, dimensions):
        tmp = np.zeros(dimensions).astype('float32')
        tmp[x,:] = 1
        return tmp.reshape(1, dimensions[0], dimensions[1])
    def numerically_safe_sample(probs):
        uniform = np.random.uniform(0,1)
        cdf = 0
        for index,p in enumerate(probs):
            cdf+=p
            if uniform<cdf:
              return index
        return len(probs)-1

    # We start the sentence with the start token
    new_sentence = one_hot( model.getCode(starting_string[0]), (word_dim,minibatch_dim))
    for c in starting_string[1:]:
        c_one_hot = one_hot( model.getCode(c), (word_dim,minibatch_dim))
        new_sentence = np.vstack((new_sentence,c_one_hot))
   
    # Repeat until we reach the sample limit
    h_init = np.zeros((len(model.layers), model.hidden_dim, minibatch_dim)).astype('float32')
    c_init = np.zeros((len(model.layers), model.hidden_dim, minibatch_dim)).astype('float32')

    all_but_last_letter = new_sentence[:-1,:,:].reshape(new_sentence.shape[0]-1,new_sentence.shape[1],new_sentence.shape[2])
    
    _, h_prev, c_prev = model.forward_propagation(all_but_last_letter, h_init, c_init, softmax_temp) # get probability of next character

    while len(new_sentence)<(len(starting_string)+sample_limit):
        last_letter = new_sentence[-1,:,:].reshape(1,new_sentence.shape[1],new_sentence.shape[2])

        next_char_probs, h_prev, c_prev = model.forward_propagation(last_letter, h_prev, c_prev, softmax_temp) # get probability of next character

        next_char_probs = [_ for _ in next_char_probs[-1,:,0]]
        #print [(code_to_char_dict[i],round(x,2)) for i,x in enumerate(next_char_probs)]

        if sample_from_distribution: # either sample from the distribution or take the most likely next character
            sampled_letter = numerically_safe_sample(next_char_probs)
        else:
            sampled_letter = np.argmax(next_char_probs)

        if sampled_letter not in code_to_char_dict: continue
        if code_to_char_dict[sampled_letter]=='NULL': continue

        sampled_one_hot = one_hot(sampled_letter, (word_dim,minibatch_dim)) # convert code to one-hot
        new_sentence = np.vstack((new_sentence,sampled_one_hot)) # stack this one-hot onto sentence thus far

        #if (len(new_sentence)-len(starting_string))%100==0: print (len(new_sentence)-len(starting_string))#"".join([code_to_char_dict[np.argmax(x)] for x in new_sentence[:,:,-1]])

    sentence_str = [code_to_char_dict[np.argmax(x)] for x in new_sentence[:,:,-1]]
    return sentence_str



if __name__=="__main__":
    DATAFILE = "saved_model_parameters/LSTM_savedparameters_206187.9__04-04___12-24-03.npz"
    SAMPLE_LIMIT = 2000
    SOFTMAX_TEMPERATURES = [.2,.3,.4,.5]#np.linspace(.4, 1, 3)
    NUM_SENTENCES = 1
    STARTING_STRING = '''static char *
regpiece(flagp)
int *flagp;
{
    register char *ret;
    register char op;
    register char *next;
    int flags;

    ret = regatom(&flags);
    if (ret == NULL)
        return(NULL);

    op = *regparse;
    if (!ISMULT(op)) {
        *flagp = flags;
        return(ret);
    }

    if (!(flags&HASWIDTH) && op != '?')
        FAIL("*+ operand could be empty");
    *flagp = (op != '+') ? (WORST|SPSTART) : (WORST|HASWIDTH);

    if (op == '*' && (flags&SIMPLE))
        reginsert(STAR, ret);
    else if (op == '*') {
        /* Emit x* as (x&|), where & means "self". */
        reginsert(BRANCH, ret);         /* Either x */
        regoptail(ret, regnode(BACK));      /* and loop */
        regoptail(ret, ret);            /* back */
        regtail(ret, regnode(BRANCH));      /* or */
        regtail(ret, regnode(NOTHING));     /* null. */
    } else if (op == '+' && (flags&SIMPLE))
        reginsert(PLUS, ret);
    else if (op == '+') {
        /* Emit x+ as x(&|), where & means "self". */
        next = regnode(BRANCH);         /* Either */
        regtail(ret, next);
        regtail(regnode(BACK), ret);        /* loop back */
        regtail(next, regnode(BRANCH));     /* or */
        regtail(ret, regnode(NOTHING));     /* null. */
    } else if (op == '?') {
        /* Emit x? as (x|) */
        reginsert(BRANCH, ret);         /* Either x */
        regtail(ret, regnode(BRANCH));      /* or */
        next = regnode(NOTHING);        /* null. */
        regtail(ret, next);
        regoptail(ret, next);
    }
    regparse++;
    if (ISMULT(*regparse))
        FAIL("nested *?+");

    return(ret);
}'''
    model = load_model_parameters_lstm(path=DATAFILE)

    for softmax_temp in SOFTMAX_TEMPERATURES:
        print "\n\nSoftmax temperature: {0}".format(softmax_temp)
        
        for i in range(NUM_SENTENCES):
            sent = generate_sentence(model, sample_limit=SAMPLE_LIMIT, softmax_temp=softmax_temp, starting_string=STARTING_STRING)
            print "".join(sent)
            print






