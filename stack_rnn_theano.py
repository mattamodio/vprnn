import sys
sys.path.append('rnn-tutorial-rnnlm/')
import numpy as np
import theano as theano
import theano.tensor as T
import theano.ifelse
from utils import *
import operator

theano.exception_verbosity='high'

class StackRNNTheano:
    
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4, stack_size=200, push_pop_mapping=None):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # A vector that indicates whether each symbol in our alphabet is a push/pop/internal symbol
        self.push_pop_mapping = push_pop_mapping
        # The stack must be bounded, so here is its upper-bound
        self.stack_size = stack_size
        # The dimensions to make a row of the stack for pushing/popping
        #

        # Randomly initialize the network parameters
        W_hx = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        W_yh = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        W_hh = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        # Theano: Created shared variables
        self.W_hx = theano.shared(name='W_hx', value=W_hx.astype(theano.config.floatX))
        self.W_yh = theano.shared(name='W_yh', value=W_yh.astype(theano.config.floatX))
        self.W_hh = theano.shared(name='W_hh', value=W_hh.astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

        


    def __theano_build__(self):
        W_hx, W_yh, W_hh = self.W_hx, self.W_yh, self.W_hh
        x = T.ivector('x')
        y = T.ivector('y')
        ptr = T.scalar(name='ptr', dtype='int32')
        ptr=0

        def forward_prop_step(x, stack, i, h, W_hx, W_yh, W_hh):
            stack, i, h = stack_update(x, stack, i, h)
            h = T.tanh(W_hx[:,x] + W_hh.dot(h))
            output = T.nnet.softmax(W_yh.dot(h))
            return [output[0], stack, i, h]

        def stack_update(x, stack, i, h):
            #mappedX = self.push_pop_mapping[x]
            mappedX = np.random.randint(low=-1,high=2)
            isPush, isPop = (0,0)
            if mappedX>0: isPush=1
            if mappedX<1: isPop=1



            stack_return = T.switch(isPush, T.set_subtensor(stack[i+1], h),
                                    T.switch(isPop, T.set_subtensor(stack[i], T.zeros(self.hidden_dim, theano.config.floatX)),
                                    T.set_subtensor(stack[0], stack[0])))

            i_return = T.switch(isPush, T.add(i,1),
                                T.switch(isPop, T.add(i,-1),
                                i))

            h_return= T.switch(isPush, h,
                         T.switch(isPop, stack[i],
                         h))

            return stack_return, i_return, h_return



        [output, stack, i, h], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros((self.stack_size, self.hidden_dim))), ptr, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[W_hx, W_yh, W_hh],
            truncate_gradient=self.bptt_truncate,
            strict=True)
        
        prediction = T.argmax(output, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(output, y))
        
        # Gradients
        dW_hx = T.grad(o_error, W_hx)
        dW_yh = T.grad(o_error, W_yh)
        dW_hh = T.grad(o_error, W_hh)
        
        # Assign functions
        self.forward_propagation = theano.function([x], output)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dW_hx, dW_yh, dW_hh])
        
        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [], 
                      updates=[(self.W_hx, self.W_hx - learning_rate * dW_hx),
                              (self.W_yh, self.W_yh - learning_rate * dW_yh),
                              (self.W_hh, self.W_hh - learning_rate * dW_hh)])
    
    def calculate_total_loss(self, X, Y):

        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)   

if __name__=="__main__":
    model = StackRNNTheano(100, push_pop_mapping={})
