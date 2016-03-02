import sys
import numpy as np
import theano as theano
import theano.tensor as T
from master_utils import *
import operator
from theano import tensor as T, function, printing

# theano.config.optimizer='None'
# theano.config.compute_test_value = 'raise'
# theano.config.exception_verbosity = 'high'
# #theano.config.profile = True
# theano.config.floatX = 'float32' # for GPU?

class Layer(object):
    def __init__(self, celltype, word_dim, hidden_dim, minibatch_dim, activation, want_stack=False, stack_height=None, push_vec=None, pop_vec=None):
        self.celltype = celltype
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.minibatch_dim = minibatch_dim
        self.activation = activation
        self.want_stack = want_stack
        self.stack_height = stack_height
        self.push_vec = push_vec
        self.pop_vec = pop_vec
        

        if celltype=='LSTM':
            self.initLSTM()

    def initLSTM(self):
        word_dim, hidden_dim, minibatch_dim = self.word_dim, self.hidden_dim, self.minibatch_dim
        W_x_i = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        W_h_i = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_x_o = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        W_h_o = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_x_f = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        W_h_f = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_x_g = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        W_h_g = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))

        self.W_x_i = theano.shared(name='W_x_i', value=W_x_i.astype(theano.config.floatX))
        self.W_h_i = theano.shared(name='W_h_i', value=W_h_i.astype(theano.config.floatX))
        self.W_x_o = theano.shared(name='W_x_o', value=W_x_o.astype(theano.config.floatX))
        self.W_h_o = theano.shared(name='W_h_o', value=W_h_o.astype(theano.config.floatX))
        self.W_x_f = theano.shared(name='W_x_f', value=W_x_f.astype(theano.config.floatX))
        self.W_h_f = theano.shared(name='W_h_f', value=W_h_f.astype(theano.config.floatX))
        self.W_x_g = theano.shared(name='W_x_g', value=W_x_g.astype(theano.config.floatX))
        self.W_h_g = theano.shared(name='W_h_g', value=W_h_g.astype(theano.config.floatX))

        #########################################################
        #########################################################
        if self.want_stack:


            W_h_stack_pop = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
            W_h_prev_pop = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
            # initialize 3d-stack and set ptrs to top to be first row
            stack = np.zeros((minibatch_dim, self.stack_height, hidden_dim))
            ptrs_to_top = np.zeros((minibatch_dim, self.stack_height, hidden_dim))
            ptrs_to_top[:,0,:] = 1

            self.W_h_stack_pop = theano.shared(name="W_h_stack_pop", value=W_h_stack_pop.astype(theano.config.floatX))
            self.W_h_prev_pop = theano.shared(name="W_h_prev_pop", value=W_h_prev_pop.astype(theano.config.floatX))
            self.stack = theano.shared(name='stack', value=stack.astype(theano.config.floatX))
            self.ptrs_to_top = theano.shared(name='ptrs_to_top', value=ptrs_to_top)
        #########################################################
        #########################################################

        self.h = T.unbroadcast(T.zeros((hidden_dim, minibatch_dim)), 1)
        self.c = T.unbroadcast(T.zeros((hidden_dim, minibatch_dim)), 1)

        self.params = []
        self.params.extend([self.W_x_i,self.W_h_i,self.W_x_o,
                            self.W_h_o,self.W_x_f,self.W_h_f,
                            self.W_x_g,self.W_h_g])

        #########################################################
        #########################################################
        if self.want_stack:
            self.params.extend([self.W_h_stack_pop,self.W_h_prev_pop])
        #########################################################
        #########################################################

    def forward_prop(self, x, is_push, is_pop):
        #########################################################
        #########################################################
        if self.celltype=='LSTM' and not self.want_stack:
            i = T.nnet.hard_sigmoid( self.W_x_i.dot(x) + self.W_h_i.dot(self.h) )
            o = T.nnet.hard_sigmoid( self.W_x_o.dot(x) + self.W_h_o.dot(self.h) )
            f = T.nnet.hard_sigmoid( self.W_x_f.dot(x) + self.W_h_f.dot(self.h) )
            g = self.activation( self.W_x_g.dot(x) + self.W_h_g.dot(self.h) )

            c_t = f*self.c + i*g
            h_t = o*self.activation(c_t)

            self.c = c_t
            self.h = h_t

            return h_t
        #########################################################
        #########################################################

        #########################################################
        #########################################################
        elif self.celltype=='LSTM' and self.want_stack:
            ###########################################################################
            # Setup
            def update_stack_for_push(stack, ptrs_to_top, is_push, h):
                # update stack and pointers for push
                #pointers
                new_top_row = T.zeros((self.minibatch_dim,1,self.hidden_dim))
                shift_everything_down_one_row = ptrs_to_top[:,:-1,:]
                shifted_ptrs_to_top = T.concatenate( [new_top_row, shift_everything_down_one_row] ,1)
                is_not_push = (1-is_push)
                updated_stack_ptrs = shifted_ptrs_to_top*is_push + ptrs_to_top*is_not_push
                #stack
                h=T.transpose(h).reshape((self.minibatch_dim,1,self.hidden_dim))
                stack_updates = shifted_ptrs_to_top * h * is_push
                updated_stack_values = stack_updates + stack

                return updated_stack_values, updated_stack_ptrs

            def update_stack_for_pop(stack, ptrs_to_top, is_pop):
                # update stack and pointers for pop
                #pointers
                new_bottom_row = T.zeros((self.minibatch_dim,1,self.hidden_dim))
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
            ###########################################################################



            ###########################################################################
            postpush_stack_values, postpush_stack_ptrs = update_stack_for_push(self.stack, self.ptrs_to_top, is_push, self.h)
            postpop_stack_values, postpop_stack_ptrs, h_popped = update_stack_for_pop(postpush_stack_values, postpush_stack_ptrs, is_pop)

            self.stack = postpop_stack_values
            self.ptrs_to_top = postpop_stack_ptrs
            h_prime = self.W_h_prev_pop.dot(self.h) + self.W_h_stack_pop.dot(h_popped)

            theano.printing.Print("W_x_i shape")(self.W_x_i.shape)
            theano.printing.Print("x shape")(x.shape)
            theano.printing.Print("h_prime shape")(h_prime.shape)
            i = T.nnet.hard_sigmoid( self.W_x_i.dot(x) + self.W_h_i.dot(h_prime) )
            o = T.nnet.hard_sigmoid( self.W_x_o.dot(x) + self.W_h_o.dot(h_prime) )
            f = T.nnet.hard_sigmoid( self.W_x_f.dot(x) + self.W_h_f.dot(h_prime) )
            g = T.tanh( self.W_x_g.dot(x) + self.W_h_g.dot(h_prime) )

            c_t = f*self.c + i*g
            h_t = o*T.tanh(c_t)

            self.c = c_t
            self.h = h_t

            return h_t
        #########################################################
        #########################################################



        

class LSTMTheano:
    
    def __init__(self, word_dim, hidden_dim=100, minibatch_dim=1, bptt_truncate=4, num_layers=1, optimization='RMSprop', activation='tanh', want_stack=False, stack_height=None, push_vec=None, pop_vec=None):
        #########################################################
        #########################################################
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.minibatch_dim = minibatch_dim
        self.bptt_truncate = bptt_truncate
        self.optimization = optimization
        #########################################################
        #########################################################
        # Activations
        def relu(x):
            return x * (x>0)
        if activation=='tanh': self.activation = T.tanh
        elif activation=='relu': self.activation = relu
        #########################################################
        #########################################################
        # Parameters
        self.layers = []
        self.params = []

        initial_layer = Layer('LSTM', word_dim, hidden_dim, minibatch_dim, activation, want_stack, stack_height, push_vec, pop_vec)
        self.layers.append(initial_layer)
        self.params.extend(initial_layer.params)
        for _ in xrange(1,num_layers):
            inner_layer = Layer('LSTM', hidden_dim, hidden_dim, minibatch_dim, activation, want_stack, stack_height, push_vec, pop_vec)
            self.layers.append(inner_layer)
            self.params.extend(inner_layer.params)
        
        W_hy = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (word_dim, hidden_dim))
        self.W_hy = theano.shared(name='W_hy', value=W_hy.astype(theano.config.floatX))
        self.params.append(self.W_hy)

       
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)


        if push_vec!=None:
            PUSH = push_vec.reshape((word_dim, minibatch_dim))
            self.PUSH = theano.shared(name='PUSH', value=PUSH.astype(theano.config.floatX))
        
        if pop_vec!=None:
            POP = pop_vec.reshape((word_dim, minibatch_dim))
            self.POP = theano.shared(name='POP', value=POP.astype(theano.config.floatX))

        #########################################################
        #########################################################
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
        #########################################################
        #########################################################
    
    def __theano_build__(self):
        # theano variables
        x = T.tensor3('x')
        y = T.tensor3('y')
        learning_rate = T.scalar('learning_rate')

        self.sequence_length = 10 # for creating test data
        x.tag.test_value = np.random.uniform(0, 1, size=(self.sequence_length,self.word_dim,self.minibatch_dim)).astype('float64')
        y.tag.test_value = np.random.uniform(0, 1, size=(self.sequence_length,self.word_dim,self.minibatch_dim)).astype('float64')
        learning_rate.tag.test_value = .01
   

        def forward_prop_step(x_t):
            BroadcastToAllHiddenDims_Type = T.TensorType('float64', (True,True,True))
            is_push = BroadcastToAllHiddenDims_Type()
            is_pop = BroadcastToAllHiddenDims_Type()
            # Map input to {push,pop,internal}
            argm_xt = T.argmax(x_t, axis=0)
            argm_push = T.argmax(self.PUSH, axis=0)
            argm_pop = T.argmax(self.POP, axis=0)
            is_push = T.eq(argm_xt, argm_push)
            is_pop = T.eq(argm_xt, argm_pop)


            is_push = is_push.reshape((self.minibatch_dim,1,1))
            is_pop = is_pop.reshape((self.minibatch_dim,1,1))

            layer_input = x_t
            for layer in self.layers:
                theano.printing.Print('layer_input.shape')(layer_input.shape)
                layer_input = layer.forward_prop(layer_input, is_push, is_pop)

            o_t = T.transpose( T.nnet.softmax( T.transpose(self.W_hy.dot(layer_input)) ) )
            

            return o_t

        o, updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None],
            truncate_gradient=self.bptt_truncate)
        
        theano.printing.Print('o.shape')(o.shape)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        

        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.ce_error = theano.function([x, y], o_error)

        

        ###########################################################################
        ###########################################################################
        # Optimization methods
        def SGD(cost, params, learning_rate):
            grads = T.grad(cost=cost, wrt=params)
            updates = []
            for p,g in zip(params, grads):
                update = (p, p - learning_rate*g)
                updates.append(update)
            return updates

        def RMSprop(cost, params, learning_rate=0.001, b1=0.9, b2=0.999, epsilon=1e-6):
            grads = T.grad(cost=cost, wrt=params)
            updates = []
            for p, g in zip(params, grads):

                m_old = theano.shared(p.get_value() * 0.)
                v_old = theano.shared(p.get_value() * 0.)

                m = b1*m_old + (1-b1)*g
                v = b2*v_old + (1-b2)*(g**2)

                update = p - (learning_rate*m) / (T.sqrt(T.sqrt(v)) + epsilon)

                updates.append((m_old, m))
                updates.append((v_old, v))
                updates.append((p, update))

            return updates

        if self.optimization=='RMSprop':
            updates = RMSprop(o_error, self.params, learning_rate=learning_rate)
        elif self.optimization=='SGD':
            updates = SGD(o_error, self.params, learning_rate=learning_rate)


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

