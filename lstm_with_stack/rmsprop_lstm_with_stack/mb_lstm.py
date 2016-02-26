import sys
import numpy as np
import theano as theano
import theano.tensor as T
from rmsprop_utils_lstm_with_stack import *
import operator
from theano import tensor as T, function, printing, typed_list
from theano.ifelse import ifelse

# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'
# theano.config.compute_test_value = 'raise'
# theano.config.profile = True

class LSTM_with_stack:
    
    def __init__(self, word_dim, hidden_dim=100, minibatch_size=1, bptt_truncate=-1, sequence_length=10, push_vec=None, pop_vec=None, stack_height=20):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.minibatch_size = minibatch_size
        self.bptt_truncate = bptt_truncate
        self.sequence_length = sequence_length
        self.stack_height = stack_height


        if push_vec==None:
            PUSH = np.zeros((word_dim, minibatch_size))
            PUSH[0] = 1
            self.PUSH = theano.shared(name='PUSH', value=PUSH.astype(theano.config.floatX))
        else:
            PUSH = push_vec.reshape((word_dim, minibatch_size))
            self.PUSH = theano.shared(name='PUSH', value=PUSH.astype(theano.config.floatX))

        if pop_vec==None:
            POP = np.zeros((word_dim, minibatch_size))
            POP[1] = 1
            self.POP = theano.shared(name='POP', value=POP.astype(theano.config.floatX))
        else:
            POP = pop_vec.reshape((word_dim, minibatch_size))
            self.POP = theano.shared(name='POP', value=POP.astype(theano.config.floatX))

        
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

        W_x_i_2 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_h_i_2 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_x_o_2 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_h_o_2 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_x_f_2 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_h_f_2 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_x_g_2 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_h_g_2 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))

        #W_h_push = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_h_stack_pop = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_h_prev_pop = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        # initialize 3d-stack and set ptrs to top to be first row
        stack = np.zeros((minibatch_size, stack_height, hidden_dim))
        ptrs_to_top = np.zeros((minibatch_size, stack_height, hidden_dim))
        ptrs_to_top[:,0,:] = 1


        W_h_stack_pop_2 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_h_prev_pop_2 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        # initialize 3d-stack and set ptrs to top to be first row
        stack_2 = np.zeros((minibatch_size, stack_height, hidden_dim))
        ptrs_to_top_2 = np.zeros((minibatch_size, stack_height, hidden_dim))
        ptrs_to_top_2[:,0,:] = 1

        # # for testing
        # stack[0,1,:] = 1
        # stack[0,1,0] = .5
        # ptrs_to_top[0,1,:] = 1
        # ptrs_to_top[0,0,:] = 0
        # stack_2[0,1,:] = 1
        # stack_2[0,1,0] = .5
        # ptrs_to_top_2[0,1,:] = 1
        # ptrs_to_top_2[0,0,:] = 0

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

        self.W_x_i_2 = theano.shared(name='W_x_i_2', value=W_x_i_2.astype(theano.config.floatX))
        self.W_h_i_2 = theano.shared(name='W_h_i_2', value=W_h_i_2.astype(theano.config.floatX))
        self.W_x_o_2 = theano.shared(name='W_x_o_2', value=W_x_o_2.astype(theano.config.floatX))
        self.W_h_o_2 = theano.shared(name='W_h_o_2', value=W_h_o_2.astype(theano.config.floatX))
        self.W_x_f_2 = theano.shared(name='W_x_f_2', value=W_x_f_2.astype(theano.config.floatX))
        self.W_h_f_2 = theano.shared(name='W_h_f_2', value=W_h_f_2.astype(theano.config.floatX))
        self.W_x_g_2 = theano.shared(name='W_x_g_2', value=W_x_g_2.astype(theano.config.floatX))
        self.W_h_g_2 = theano.shared(name='W_h_g_2', value=W_h_g_2.astype(theano.config.floatX))


        #self.W_h_push = theano.shared(name="W_h_push", value=W_h_push.astype(theano.config.floatX))
        self.W_h_stack_pop = theano.shared(name="W_h_stack_pop", value=W_h_stack_pop.astype(theano.config.floatX))
        self.W_h_prev_pop = theano.shared(name="W_h_prev_pop", value=W_h_prev_pop.astype(theano.config.floatX))
        self.stack = theano.shared(name='stack', value=stack.astype(theano.config.floatX))
        self.ptrs_to_top = theano.shared(name='ptrs_to_top', value=ptrs_to_top)

        self.W_h_stack_pop_2 = theano.shared(name="W_h_stack_pop_2", value=W_h_stack_pop_2.astype(theano.config.floatX))
        self.W_h_prev_pop_2 = theano.shared(name="W_h_prev_pop_2", value=W_h_prev_pop_2.astype(theano.config.floatX))
        self.stack_2 = theano.shared(name='stack_2', value=stack.astype(theano.config.floatX))
        self.ptrs_to_top_2 = theano.shared(name='ptrs_to_top_2', value=ptrs_to_top_2)


        
        self.params = []
        self.params.extend([self.W_x_i,self.W_h_i,self.W_x_o,
                            self.W_h_o,self.W_x_f,self.W_h_f,
                            self.W_x_g,self.W_h_g,
                            self.W_x_i_2,self.W_h_i_2,self.W_x_o_2,
                            self.W_h_o_2,self.W_x_f_2,self.W_h_f_2,
                            self.W_x_g_2,self.W_h_g_2,
                            self.W_hy,
                            self.W_h_stack_pop,self.W_h_prev_pop,
                            self.W_h_stack_pop_2,self.W_h_prev_pop_2])
        #self.params.append(self.W_h_push)
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        learning_rate = T.scalar('learning_rate')


        # x.tag.test_value = np.random.uniform(0, 1, size=(10,self.word_dim,self.minibatch_size)).astype('float64')
        # y.tag.test_value = np.random.uniform(0, 1, size=(10,self.word_dim,self.minibatch_size)).astype('float64')
        # learning_rate.tag.test_value = .001
        # x.tag.test_value[0,2,1] = 1
        # x.tag.test_value[0,29,0] = 1

        def update_stack_for_push(stack, ptrs_to_top, is_push, h):
            # update stack and pointers for push
            #pointers
            new_top_row = T.zeros((self.minibatch_size,1,self.hidden_dim))
            shift_everything_down_one_row = ptrs_to_top[:,:-1,:]
            shifted_ptrs_to_top = T.concatenate( [new_top_row, shift_everything_down_one_row] ,1)
            is_not_push = (1-is_push)
            updated_stack_ptrs = shifted_ptrs_to_top*is_push + ptrs_to_top*is_not_push
            #stack
            h=T.transpose(h).reshape((self.minibatch_size,1,self.hidden_dim))
            stack_updates = shifted_ptrs_to_top * h * is_push
            updated_stack_values = stack_updates + stack

            return updated_stack_values, updated_stack_ptrs

        def update_stack_for_pop(stack, ptrs_to_top, is_pop):
            # update stack and pointers for pop
            #pointers
            new_bottom_row = T.zeros((self.minibatch_size,1,self.hidden_dim))
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


        def forward_prop_step(x_t, h_t_prev, h_t_2_prev, c_t_2_prev, c_t_prev):

            # h_t_prev.tag.test_value = np.random.uniform(0,1, (self.hidden_dim,self.minibatch_size)).astype('float64')
            # c_t_prev.tag.test_value = np.random.uniform(0,1, (self.hidden_dim,self.minibatch_size)).astype('float64')

            ###########################################################################
            # Setup
            BroadcastToAllHiddenDims_Type = T.TensorType('float64', (True,True,True))
            is_push = BroadcastToAllHiddenDims_Type()
            is_pop = BroadcastToAllHiddenDims_Type()
            # Map input to {push,pop,internal}
            argm_xt = T.argmax(x_t, axis=0)
            argm_push = T.argmax(self.PUSH, axis=0)
            argm_pop = T.argmax(self.POP, axis=0)
            is_push = T.eq(argm_xt, argm_push)
            is_pop = T.eq(argm_xt, argm_pop)


            is_push = is_push.reshape((self.minibatch_size,1,1))
            is_pop = is_pop.reshape((self.minibatch_size,1,1))
            ###########################################################################



            ###########################################################################
            # Layer 1
            postpush_stack_values, postpush_stack_ptrs = update_stack_for_push(self.stack, self.ptrs_to_top, is_push, h_t_prev)
            postpop_stack_values, postpop_stack_ptrs, h_popped = update_stack_for_pop(postpush_stack_values, postpush_stack_ptrs, is_pop)

            self.stack = postpop_stack_values
            self.ptrs_to_top = postpop_stack_ptrs
            h_prime = self.W_h_prev_pop.dot(h_t_prev) + self.W_h_stack_pop.dot(h_popped)


            i = T.nnet.hard_sigmoid( self.W_x_i.dot(x_t) + self.W_h_i.dot(h_prime) )
            o = T.nnet.hard_sigmoid( self.W_x_o.dot(x_t) + self.W_h_o.dot(h_prime) )
            f = T.nnet.hard_sigmoid( self.W_x_f.dot(x_t) + self.W_h_f.dot(h_prime) )
            g = T.tanh( self.W_x_g.dot(x_t) + self.W_h_g.dot(h_prime) )

            c_t = f*c_t_prev + i*g
            h_t = o*T.tanh(c_t)
            ###########################################################################



            ###########################################################################
            # Layer 2
            postpush_stack_values_2, postpush_stack_ptrs_2 = update_stack_for_push(self.stack_2, self.ptrs_to_top_2, is_push, h_t_2_prev)
            postpop_stack_values_2, postpop_stack_ptrs_2, h_popped_2 = update_stack_for_pop(postpush_stack_values_2, postpush_stack_ptrs_2, is_pop)

            self.stack_2 = postpop_stack_values_2
            self.ptrs_to_top_2 = postpop_stack_ptrs_2
            h_prime_2 = self.W_h_prev_pop_2.dot(h_t_2_prev) + self.W_h_stack_pop_2.dot(h_popped_2)

            i_2 = T.nnet.hard_sigmoid( self.W_x_i_2.dot(h_t) + self.W_h_i_2.dot(h_prime_2) )
            o_2 = T.nnet.hard_sigmoid( self.W_x_o_2.dot(h_t) + self.W_h_o_2.dot(h_prime_2) )
            f_2 = T.nnet.hard_sigmoid( self.W_x_f_2.dot(h_t) + self.W_h_f_2.dot(h_prime_2) )
            g_2 = T.tanh( self.W_x_g_2.dot(h_t) + self.W_h_g_2.dot(h_prime_2) )

            c_t_2 = f_2*c_t_2_prev + i_2*g_2
            h_t_2 = o_2*T.tanh(c_t_2)
            ###########################################################################


            ###########################################################################
            # # Output
            o_t = T.transpose( T.nnet.softmax( T.transpose(self.W_hy.dot(h_t_2)) ) )
            # o_t = T.transpose( T.nnet.softmax( T.transpose(self.W_hy.dot(h_t)) ) )
            # h_t_2 = h_t_2_prev
            # c_t_2 = c_t_2_prev
            ###########################################################################

            return [o_t, h_t, h_t_2, c_t, c_t_2]



        h_init = dict(initial=T.unbroadcast(T.zeros((self.hidden_dim, self.minibatch_size)), 1))
        h2_init = dict(initial=T.unbroadcast(T.zeros((self.hidden_dim, self.minibatch_size)), 1))
        c_init = dict(initial=T.unbroadcast(T.zeros((self.hidden_dim, self.minibatch_size)), 1))
        c2_init = dict(initial=T.unbroadcast(T.zeros((self.hidden_dim, self.minibatch_size)), 1))

        [o,h,h2,c,c2], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, h_init, h2_init, c_init, c2_init],
            truncate_gradient=self.bptt_truncate)
        


        # def error_last_x_steps(o,y,x):
        #     error = 0
        #     for i in xrange(1,x+1):
        #         error += T.sum(T.nnet.categorical_crossentropy(o[:][:][-i], y[:][:][-i]))
        #     return error

        #o_error = error_last_x_steps(o,y, self.sequence_length/2)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        #o_error = T.sum(T.nnet.categorical_crossentropy(o[:][:][-1], y[:][:][-1]))
        
        
        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.ce_error = theano.function([x, y], o_error)

        ###########################################################################
        ###########################################################################
        def RMSprop(cost, params, learning_rate=0.001, b1=0.9, b2=.999, epsilon=1e-6):
            grads = T.grad(cost=cost, wrt=params)
            updates = []
            for p, g in zip(params, grads):
                m_old = theano.shared(p.get_value() * 0.)
                v_old = theano.shared(p.get_value() * 0.)

                m = b1*m_old + (1-b1)*g
                v = b2*v_old + (1-b2)*(g**2)

                update = p - (learning_rate*m) / (T.sqrt(v) + epsilon)

                updates.append((m_old, m))
                updates.append((v_old, v))
                updates.append((p, update))

            return updates

        updates = RMSprop(o_error, self.params, learning_rate=learning_rate)


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

