import sys
import numpy as np
import theano as theano
import theano.tensor as T
from rmsprop_utils import *
import operator
from theano import tensor as T, function, printing, typed_list

# theano.config.optimizer='None'
# theano.config.compute_test_value = 'raise'

class LSTMTheano:
    
    def __init__(self, word_dim, hidden_dim=100, minibatch_size=1, bptt_truncate=4, momentum=0):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.minibatch_size = minibatch_size
        self.bptt_truncate = bptt_truncate
        self.mom = momentum
        

        # Randomly initialize the network parameters
        W_x_i = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        W_h_i = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_x_o = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        W_h_o = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_x_f = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        W_h_f = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_x_g = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        W_h_g = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_hy = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (word_dim, hidden_dim))

        # Theano: Created shared variables
        self.W_x_i = theano.shared(name='W_x_i', value=W_x_i.astype(theano.config.floatX))
        self.W_h_i = theano.shared(name='W_h_i', value=W_h_i.astype(theano.config.floatX))
        self.W_x_o = theano.shared(name='W_x_o', value=W_x_o.astype(theano.config.floatX))
        self.W_h_o = theano.shared(name='W_h_o', value=W_h_o.astype(theano.config.floatX))
        self.W_x_f = theano.shared(name='W_x_f', value=W_x_f.astype(theano.config.floatX))
        self.W_h_f = theano.shared(name='W_h_f', value=W_h_f.astype(theano.config.floatX))
        self.W_x_g = theano.shared(name='W_x_g', value=W_x_g.astype(theano.config.floatX))
        self.W_h_g = theano.shared(name='W_h_g', value=W_h_g.astype(theano.config.floatX))
        self.W_hy = theano.shared(name='W_hy', value=W_hy.astype(theano.config.floatX))  

        self.params = []
        self.params.extend([self.W_x_i,self.W_h_i,self.W_x_o,self.W_h_o,self.W_x_f,self.W_h_f,self.W_x_g,self.W_h_g,self.W_hy]) 
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        W_x_i,W_h_i,W_x_o,W_h_o,W_x_f,W_h_f,W_x_g,W_h_g,W_hy = self.W_x_i,self.W_h_i,self.W_x_o,self.W_h_o,self.W_x_f,self.W_h_f,self.W_x_g,self.W_h_g,self.W_hy
        
        # theano variables
        x = T.tensor3('x')
        y = T.tensor3('y')
        learning_rate = T.scalar('learning_rate')

        def forward_prop_step(x_t, h_t_prev, c_t_prev):
            i = T.nnet.hard_sigmoid( W_x_i.dot(x_t) + W_h_i.dot(h_t_prev) )
            o = T.nnet.hard_sigmoid( W_x_o.dot(x_t) + W_h_o.dot(h_t_prev) )
            f = T.nnet.hard_sigmoid( W_x_f.dot(x_t) + W_h_f.dot(h_t_prev) )
            g = T.tanh( W_x_g.dot(x_t) + W_h_g.dot(h_t_prev) )

            c_t = f*c_t_prev + i*g
            h_t = o*T.tanh(c_t)

            o_t = T.transpose( T.nnet.softmax( T.transpose(W_hy.dot(h_t)) ) )
            return [o_t, h_t, c_t]


        h_init = dict(initial=T.unbroadcast(T.zeros((self.hidden_dim, self.minibatch_size)), 1))
        c_init = dict(initial=T.unbroadcast(T.zeros((self.hidden_dim, self.minibatch_size)), 1))

        [o,h,c], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, h_init, c_init],
            truncate_gradient=self.bptt_truncate)
        
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Gradients
        dW_x_i = T.grad(o_error, W_x_i)
        dW_h_i = T.grad(o_error, W_h_i)
        dW_x_o = T.grad(o_error, W_x_o)
        dW_h_o = T.grad(o_error, W_h_o)
        dW_x_f = T.grad(o_error, W_x_f)
        dW_h_f = T.grad(o_error, W_h_f)
        dW_x_g = T.grad(o_error, W_x_g)
        dW_h_g = T.grad(o_error, W_h_g)
        dW_hy = T.grad(o_error, W_hy)

        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.ce_error = theano.function([x, y], o_error)

        # SGD
        self.sgd_step = theano.function([x,y,learning_rate], [], 
                      updates=[(self.W_x_i, self.W_x_i - learning_rate * dW_x_i),
                              (self.W_h_i, self.W_h_i - learning_rate * dW_h_i),
                              (self.W_x_o, self.W_x_o - learning_rate * dW_x_o),
                              (self.W_h_o, self.W_h_o - learning_rate * dW_h_o),
                              (self.W_x_f, self.W_x_f - learning_rate * dW_x_f),
                              (self.W_h_f, self.W_h_f - learning_rate * dW_h_f),
                              (self.W_x_g, self.W_x_g - learning_rate * dW_x_g),
                              (self.W_h_g, self.W_h_g - learning_rate * dW_h_g),
                              (self.W_hy, self.W_hy - learning_rate * dW_hy)])

        ###########################################################################
        ###########################################################################
        gparams = []
        for param in self.params:
            gparam = T.grad(o_error, param)
            gparams.append(gparam)

        updates = {}
        for param, gparam in zip(self.params, gparams):
            weight_update = self.updates[param]
            upd = self.mom * weight_update - learning_rate * gparam
            updates[weight_update] = upd
            updates[param] = param + upd

        self.train_model = theano.function(inputs=[x,y,learning_rate],
                                      outputs=[],
                                      updates=updates)
        ###########################################################################
        ###########################################################################
    
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)

if __name__=="__main__":
    model = LSTMTheano(100)

