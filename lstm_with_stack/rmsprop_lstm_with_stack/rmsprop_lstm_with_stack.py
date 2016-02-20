import sys
import numpy as np
import theano as theano
import theano.tensor as T
from rmsprop_utils_lstm_with_stack import *
import operator
from theano import tensor as T, function, printing, typed_list
from theano.ifelse import ifelse
from theano.printing import pydotprint

# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'
# theano.config.compute_test_value = 'raise'
# theano.config.profile = True

class LSTM_with_stack:
    
    def __init__(self, word_dim, hidden_dim=100, minibatch_size=1, bptt_truncate=-1, push_vec=None, pop_vec=None):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.minibatch_size = minibatch_size
        self.bptt_truncate = bptt_truncate

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

        W_h_push = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_h_stack_pop = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        W_h_prev_pop = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        stack = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, minibatch_size, 15))
        ptr_to_top = 0

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
        self.ptr_to_top = theano.shared(name='ptr_to_top', value=ptr_to_top)


        
        self.params = []
        self.params.extend([self.W_x_i,self.W_h_i,self.W_x_o,self.W_h_o,self.W_x_f,self.W_h_f,self.W_x_g,self.W_h_g,\
                            self.W_x_i_2,self.W_h_i_2,self.W_x_o_2,self.W_h_o_2,self.W_x_f_2,self.W_h_f_2,self.W_x_g_2,\
                            self.W_h_g_2,self.W_hy,self.W_h_stack_pop,self.W_h_prev_pop])
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
        W_x_i,W_h_i,W_x_o,W_h_o,W_x_f,W_h_f,W_x_g,W_h_g,W_hy = self.W_x_i,self.W_h_i,self.W_x_o,self.W_h_o,self.W_x_f,self.W_h_f,self.W_x_g,self.W_h_g,self.W_hy
        
        W_x_i_2,W_h_i_2,W_x_o_2,W_h_o_2,W_x_f_2,W_h_f_2,W_x_g_2,W_h_g_2 = self.W_x_i_2,self.W_h_i_2,self.W_x_o_2,self.W_h_o_2,self.W_x_f_2,self.W_h_f_2,self.W_x_g_2,self.W_h_g_2
        

        W_h_prev_pop, W_h_stack_pop = self.W_h_prev_pop, self.W_h_stack_pop
        #W_h_push = self.W_h_push

        x = T.tensor3('x')#, dtype='float64')
        y = T.tensor3('y')#, dtype='float64')
        learning_rate = T.scalar('learning_rate')


        # x.tag.test_value = np.random.uniform(0, 1, size=(5,self.word_dim,self.minibatch_size)).astype('float64')
        # y.tag.test_value = np.random.uniform(0, 1, size=(5,self.word_dim,self.minibatch_size)).astype('float64')
        # learning_rate.tag.test_value = .001

        def forward_prop_step(x_t, h_t_prev, h_t_2_prev, c_t_2_prev, c_t_prev):

            # h_t_prev.tag.test_value = np.random.uniform(0,1, (self.hidden_dim,self.minibatch_size)).astype('float64')
            # c_t_prev.tag.test_value = np.random.uniform(0,1, (self.hidden_dim,self.minibatch_size)).astype('float64')


            argm_xt = T.argmax(x_t, axis=0)[0]
            argm_push = T.argmax(self.PUSH, axis=0)[0]
            argm_pop = T.argmax(self.POP, axis=0)[0]
            is_push = T.eq(argm_xt, argm_push)
            is_pop = T.eq(argm_xt, argm_pop)


            candidate_to_push = h_t_prev
            pushed_stack = T.set_subtensor(self.stack[:,:,self.ptr_to_top+1], candidate_to_push)


            top_of_stack = self.stack[:,:,self.ptr_to_top]
            candidate_to_pop = T.tanh( W_h_prev_pop.dot(h_t_prev) + W_h_stack_pop.dot(top_of_stack) )

            self.stack = ifelse( is_push,
                            pushed_stack,
                            ifelse( is_pop,
                                    self.stack,
                                    self.stack
                                    )
                            )

            self.ptr_to_top = ifelse( is_push,
                                 self.ptr_to_top+1,
                                 ifelse( is_pop,
                                         self.ptr_to_top-1,
                                         self.ptr_to_top
                                         )
                                 )

            h_prime = ifelse( is_push,
                              h_t_prev,
                              ifelse( is_pop,
                                      candidate_to_pop,
                                      h_t_prev
                                      )
                              )

            i = T.nnet.hard_sigmoid( W_x_i.dot(x_t) + W_h_i.dot(h_prime) )
            o = T.nnet.hard_sigmoid( W_x_o.dot(x_t) + W_h_o.dot(h_prime) )
            f = T.nnet.hard_sigmoid( W_x_f.dot(x_t) + W_h_f.dot(h_prime) )
            g = T.tanh( W_x_g.dot(x_t) + W_h_g.dot(h_prime) )

            c_t = f*c_t_prev + i*g
            h_t = o*T.tanh(c_t)

            i_2 = T.nnet.hard_sigmoid( W_x_i_2.dot(h_t_2_prev) + W_h_i_2.dot(h_t) )
            o_2 = T.nnet.hard_sigmoid( W_x_o_2.dot(h_t_2_prev) + W_h_o_2.dot(h_t) )
            f_2 = T.nnet.hard_sigmoid( W_x_f_2.dot(h_t_2_prev) + W_h_f_2.dot(h_t) )
            g_2 = T.tanh( W_x_g_2.dot(h_t_2_prev) + W_h_g_2.dot(h_t) )

            c_t_2 = f_2*c_t_2_prev + i_2*g_2
            h_t_2 = o_2*T.tanh(c_t_2)


            o_t = T.transpose( T.nnet.softmax( T.transpose(W_hy.dot(h_t_2)) ) )

            #theano.printing.Print('o_t')(o_t)

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
        

        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        
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

