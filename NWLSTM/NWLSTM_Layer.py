import sys, operator
import numpy as np
import theano
from theano import tensor as T, function, printing
from utils import *



class NWLSTM_Layer(object):
    def __init__(self, layer_num, word_dim, hidden_dim, minibatch_dim, activation,
        want_stack=False, stack_height=None, push_vec=None, pop_vec=None, null_vec=None):
        #########################################################
        #########################################################
        # Assign instance variables
        self.layer_num = layer_num
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.minibatch_dim = minibatch_dim
        self.activation = activation
        self.want_stack = want_stack
        self.stack_height = stack_height
        self.push_vec = push_vec
        self.pop_vec = pop_vec
        self.null_vec = null_vec
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
        B_f = np.random.uniform(.99, 1., (hidden_dim, 1)) #initialize forget gate close to one, encouraging early memory
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
        #########################################################
        # For stack, also create these symbolic variables
        if self.want_stack:

            W_h_stack_pop = np.random.uniform(-.01, .01, (hidden_dim, hidden_dim))
            W_h_prev_pop = np.random.uniform(-.01, .01, (hidden_dim, hidden_dim))
            # initialize 3d-stack and set ptrs to top to be first row
            stack = np.zeros((minibatch_dim, self.stack_height, hidden_dim))
            ptrs_to_top = np.zeros((minibatch_dim, self.stack_height, hidden_dim))
            ptrs_to_top[:,0,:] = 1

            self.W_h_stack_pop = theano.shared(name="W_h_stack_pop"+str(self.layer_num), value=W_h_stack_pop.astype(theano.config.floatX))
            self.W_h_prev_pop = theano.shared(name="W_h_prev_pop"+str(self.layer_num), value=W_h_prev_pop.astype(theano.config.floatX))
            self.stack = theano.shared(name='stack'+str(self.layer_num), value=stack.astype(theano.config.floatX))
            self.ptrs_to_top = theano.shared(name='ptrs_to_top'+str(self.layer_num), value=ptrs_to_top.astype(theano.config.floatX))
        #########################################################
        #########################################################
        # Group parameters together
        self.params = []
        self.params.extend([self.W_x_i,self.W_h_i,self.B_i,
                            self.W_x_o,self.W_h_o,self.B_o,
                            self.W_x_f,self.W_h_f,self.B_f,
                            self.W_x_g,self.W_h_g,self.B_g])

        if self.want_stack:
            self.params.extend([self.W_h_stack_pop,self.W_h_prev_pop])
        #########################################################
        #########################################################

    def forward_prop(self, x, h_prev, c_prev, is_push, is_pop, is_null):
        #########################################################
        #########################################################
        # Perform push/pops as necessary, updating stack and stack pointers
        if self.want_stack:
            postpush_stack_values, postpush_stack_ptrs = update_stack_for_push(self.stack, self.ptrs_to_top, is_push, h_prev)
            postpop_stack_values, postpop_stack_ptrs, h_popped = update_stack_for_pop(postpush_stack_values, postpush_stack_ptrs, is_pop)

            self.stack = postpop_stack_values
            self.ptrs_to_top = postpop_stack_ptrs
            h_prime = self.W_h_prev_pop.dot(h_prev) + self.W_h_stack_pop.dot(h_popped)
        else:
            h_prime = h_prev
        #########################################################
        #########################################################
        # Internal LSTM calculations
        i = T.nnet.hard_sigmoid( self.W_x_i.dot(x) + self.W_h_i.dot(h_prime) + self.B_i )
        o = T.nnet.hard_sigmoid( self.W_x_o.dot(x) + self.W_h_o.dot(h_prime) + self.B_o )
        f = T.nnet.sigmoid( self.W_x_f.dot(x) + self.W_h_f.dot(h_prime) + self.B_f )
        g = self.activation( self.W_x_g.dot(x) + self.W_h_g.dot(h_prime) + self.B_g )

        c = f*c_prev + i*g
        # c = c_prev + i*g
        h = o*self.activation(c_prev)

        if self.want_stack:
            h = T.switch(T.eq(is_null,1), h*0., h)
            c = T.switch(T.eq(is_null,1), c*0., c)
            return h, c
        else:
            return h, c
        #########################################################
        #########################################################

def update_stack_for_push(stack, ptrs_to_top, is_push, h):
    # update stack and pointers for push
    #pointers
    new_top_row = T.zeros((stack.shape[0],1,stack.shape[2]))
    shift_everything_down_one_row = ptrs_to_top[:,:-1,:]
    shifted_ptrs_to_top = T.concatenate( [new_top_row, shift_everything_down_one_row] ,1)
    is_not_push = (1-is_push)
    updated_stack_ptrs = shifted_ptrs_to_top*is_push + ptrs_to_top*is_not_push
    #stack
    h=T.transpose(h).reshape((stack.shape[0],1,stack.shape[2]))
    stack_updates = shifted_ptrs_to_top * h * is_push
    updated_stack_values = stack_updates + stack

    return updated_stack_values, updated_stack_ptrs

def update_stack_for_pop(stack, ptrs_to_top, is_pop):
    # update stack and pointers for pop
    #pointers
    new_bottom_row = T.zeros((stack.shape[0],1,stack.shape[2]))
    shift_everything_up_one_row = ptrs_to_top[:,1:,:]
    shifted_stack_ptrs = T.concatenate( [shift_everything_up_one_row, new_bottom_row] ,1)
    is_not_pop = (1-is_pop)
    updated_stack_ptrs = shifted_stack_ptrs*is_pop + ptrs_to_top*is_not_pop

    #stack
    updated_stack_values = (1-ptrs_to_top*is_pop)*stack

    #popped value for h
    h_popped = (ptrs_to_top*is_pop)*stack
    h_popped = T.transpose(T.sum(h_popped, axis=1))

    return updated_stack_values, updated_stack_ptrs, h_popped






