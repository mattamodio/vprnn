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

    def one_hot(x, dimensions):
        tmp = np.zeros(dimensions).astype('float32')
        tmp[x,:] = 1
        return tmp.reshape(1, dimensions[0], dimensions[1])

    char_to_code_dict, code_to_char_dict, word_dim, minibatch_dim = model.char_to_code_dict, model.code_to_char_dict, model.word_dim, model.minibatch_dim
    
    # We start the sentence with the start token
    new_sentence = one_hot(char_to_code_dict[starting_string[0]], (word_dim,minibatch_dim))
    for c in starting_string[1:]:
        c_one_hot = one_hot(char_to_code_dict[c], (word_dim,minibatch_dim))
        new_sentence = np.vstack((new_sentence,c_one_hot))
   
    # Repeat until we reach the sample limit
    h_init = np.zeros((len(model.layers), model.hidden_dim, minibatch_dim)).astype('float32')
    c_init = np.zeros((len(model.layers), model.hidden_dim, minibatch_dim)).astype('float32')

    all_but_last_letter = new_sentence[:-1,:,:].reshape(new_sentence.shape[0]-1,new_sentence.shape[1],new_sentence.shape[2])
    _, h_prev, c_prev = model.forward_propagation(all_but_last_letter, h_init, c_init, softmax_temp) # get probability of next character
            
    while len(new_sentence)<(len(starting_string)+sample_limit):
        try:
            last_letter = new_sentence[-1,:,:].reshape(1,new_sentence.shape[1],new_sentence.shape[2])
            next_char_probs, h_prev, c_prev = model.forward_propagation(last_letter, h_prev, c_prev, softmax_temp) # get probability of next character
            next_char_probs = [_ for _ in next_char_probs[-1,:,0]]
            #print [(code_to_char_dict[i],round(x,2)) for i,x in enumerate(next_char_probs)]

            if sample_from_distribution: # either sample from the distribution or take the most likely next character
                samples = np.random.multinomial(1, next_char_probs)
                sampled_letter = np.argmax(samples)
            else:
                sampled_letter = np.argmax(next_char_probs)

            if sampled_letter not in code_to_char_dict: continue
            if code_to_char_dict[sampled_letter]=='NULL_TOKEN': continue

            sampled_one_hot = one_hot(sampled_letter, (word_dim,minibatch_dim)) # convert code to one-hot
            new_sentence = np.vstack((new_sentence,sampled_one_hot)) # stack this one-hot onto sentence thus far
        except:
            print "Error while sampling..."
            break

    sentence_str = [code_to_char_dict[np.argmax(x)] for x in new_sentence[:,:,-1]]
    return sentence_str



if __name__=="__main__":
    DATAFILE = "saved_model_parameters/NWLSTM_savedparameters_13373.2__03-14___19-07-23.npz"
    SAMPLE_FROM_DISTRIBUTION = True
    SAMPLE_LIMIT = 50
    SOFTMAX_TEMPERATURES = [.4,.6,.8,1]# np.linspace(.25, 1.25, 3)
    NUM_SENTENCES = 1
    STARTING_STRING = '''"Oh, don't speak to me of Austria. Perhaps I don't understandhings, but Austria never has wished, and does not wish, for war.he is betraying us! Russia alone must save Europe. Our graciousovereign recognizes his high vocation and will be true to it. That ishe one thing I have faith in! Our good and wonderful sovereign has toerform the noblest role on earth, and he is so virtuous and noblehat God will not forsake him. He will fulfill his vocation andrush the hydra of revolution, which has become more terrible thanver in the person of this murderer and villain! We alone mustvenge the blood of the just one.... Whom, I ask you, can we relyn?... England with her commercial spirit will not and cannotnderstand the Emperor Alexander's loftiness of soul. She hasefused to evacuate Malta. She wanted to find, and still seeks, someecret motive in our actions. What answer did Novosiltsev get? None.he English have not understood and cannot understand theelf-abnegation of our Emperor who wants nothing for himself, but onlyesires the good of mankind. And what have they promised? Nothing! Andhat little they have promised they will not perform! Prussia haslways declared that Buonaparte is invincible, and that all Europes powerless before him.... And I don't believe a word that Hardenburgays, or Haugwitz either. This famous Prussian neutrality is just arap. I have faith only in God and the lofty destiny of our adoredonarch.'''


    model = load_model_parameters_lstm(path=DATAFILE)

    for softmax_temp in SOFTMAX_TEMPERATURES:
        print "\n\nSoftmax temperature: {0}".format(softmax_temp)
        
        for i in range(NUM_SENTENCES):
            sent = []
            sent = generate_sentence(model, sample_limit=SAMPLE_LIMIT, sample_from_distribution=SAMPLE_FROM_DISTRIBUTION, softmax_temp=softmax_temp, starting_string=STARTING_STRING)
            print "".join(sent)
            print






