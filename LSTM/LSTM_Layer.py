import sys, operator
import numpy as np
import theano
from theano import tensor as T, function, printing
from utils import *



class LSTM_Layer(object):
    def __init__(self, layer_num, word_dim, hidden_dim, minibatch_dim, activation):
        #########################################################
        #########################################################
        # Assign instance variables
        self.layer_num = layer_num
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.minibatch_dim = minibatch_dim
        self.activation = activation
        #########################################################
        #########################################################
        # Create weight/bias matrices and their symbolic shared variables
        word_dim, hidden_dim, minibatch_dim = self.word_dim, self.hidden_dim, self.minibatch_dim
        W_x_i = np.random.uniform(-.01, .01, (hidden_dim, word_dim))
        W_h_i = np.random.uniform(-.01, .01, (hidden_dim, hidden_dim))
        W_x_o = np.random.uniform(-.01, .01, (hidden_dim, word_dim))
        W_h_o = np.random.uniform(-.01, .01, (hidden_dim, hidden_dim))
        W_x_f = np.random.uniform(-.01, .01, (hidden_dim, word_dim))
        W_h_f = np.random.uniform(-.01, .01, (hidden_dim, hidden_dim))
        W_x_g = np.random.uniform(-.01, .01, (hidden_dim, word_dim))
        W_h_g = np.random.uniform(-.01, .01, (hidden_dim, hidden_dim))
        B_i = np.random.uniform(-.01, .01, (hidden_dim, 1))
        B_f = np.random.uniform(.99, 1.01, (hidden_dim, 1)) #initialize forget gate close to one, encouraging early memory
        B_o = np.random.uniform(-.01, .01, (hidden_dim, 1))
        B_g = np.random.uniform(-.01, .01, (hidden_dim, 1))

        self.W_x_i = theano.shared(name='W_x_i'+str(self.layer_num), value=W_x_i.astype(theano.config.floatX))
        self.W_h_i = theano.shared(name='W_h_i'+str(self.layer_num), value=W_h_i.astype(theano.config.floatX))
        self.W_x_o = theano.shared(name='W_x_o'+str(self.layer_num), value=W_x_o.astype(theano.config.floatX))
        self.W_h_o = theano.shared(name='W_h_o'+str(self.layer_num), value=W_h_o.astype(theano.config.floatX))
        self.W_x_f = theano.shared(name='W_x_f'+str(self.layer_num), value=W_x_f.astype(theano.config.floatX))
        self.W_h_f = theano.shared(name='W_h_f'+str(self.layer_num), value=W_h_f.astype(theano.config.floatX))
        self.W_x_g = theano.shared(name='W_x_g'+str(self.layer_num), value=W_x_g.astype(theano.config.floatX))
        self.W_h_g = theano.shared(name='W_h_g'+str(self.layer_num), value=W_h_g.astype(theano.config.floatX))
        self.B_i = theano.shared(name='B_i'+str(self.layer_num), value=B_i.astype(theano.config.floatX), broadcastable=(False,True)) #broadcast across minibatch
        self.B_f = theano.shared(name='B_f'+str(self.layer_num), value=B_f.astype(theano.config.floatX), broadcastable=(False,True))
        self.B_o = theano.shared(name='B_o'+str(self.layer_num), value=B_o.astype(theano.config.floatX), broadcastable=(False,True))
        self.B_g = theano.shared(name='B_g'+str(self.layer_num), value=B_g.astype(theano.config.floatX), broadcastable=(False,True))
        #########################################################
        # Group parameters together
        self.params = []
        self.params.extend([self.W_x_i,self.W_h_i,self.B_i,
                            self.W_x_o,self.W_h_o,self.B_o,
                            self.W_x_f,self.W_h_f,self.B_f,
                            self.W_x_g,self.W_h_g,self.B_g])
        #########################################################
        #########################################################

    def forward_prop(self, x, h_prev, c_prev):
        #########################################################
        #########################################################
        # Internal LSTM calculations
        i = T.nnet.hard_sigmoid( self.W_x_i.dot(x) + self.W_h_i.dot(h_prev) + self.B_i )
        o = T.nnet.hard_sigmoid( self.W_x_o.dot(x) + self.W_h_o.dot(h_prev) + self.B_o )
        f = T.nnet.sigmoid( self.W_x_f.dot(x) + self.W_h_f.dot(h_prev) + self.B_f )
        g = self.activation( self.W_x_g.dot(x) + self.W_h_g.dot(h_prev) + self.B_g )

        c = f*c_prev + i*g
        h = o*self.activation(c_prev)

        return h, c
        #########################################################
        #########################################################




