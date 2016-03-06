import sys, operator
import numpy as np
import theano
from theano import tensor as T, function, printing
from utils import *
from NWLSTM_Layer import NWLSTM_Layer
from theano.compile.nanguardmode import NanGuardMode

class NWLSTM_Net:
    def __init__(self, word_dim, hidden_dim=100, minibatch_dim=1, bptt_truncate=4, num_layers=1,
        optimization='RMSprop', activation='tanh', want_stack=False, stack_height=None, 
        push_vec=None, pop_vec=None, softmax_temperature=1, dropout=0):
        #########################################################
        #########################################################
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.minibatch_dim = minibatch_dim
        self.bptt_truncate = bptt_truncate
        self.optimization = optimization
        self.softmax_temperature = softmax_temperature
        self.dropout = dropout
        self.PUSH = T.addbroadcast(T.zeros((word_dim,1)), 1)
        self.POP = T.addbroadcast(T.zeros((word_dim,1)), 1)
        self.want_stack = want_stack
        #########################################################
        #########################################################
        # Activations
        if activation=='tanh': self.activation = T.tanh
        elif activation=='relu': self.activation = T.nnet.relu
        #########################################################
        #########################################################
        # Parameters
        self.layers = []
        self.params = []
        #########################################################
        #########################################################
        # Initialize layers and group their parameters together
        initial_layer = NWLSTM_Layer(layer_num=1, word_dim=word_dim,
            hidden_dim=hidden_dim, minibatch_dim=minibatch_dim, 
            activation=self.activation, want_stack=want_stack,
            stack_height=stack_height, push_vec=push_vec, pop_vec=pop_vec)
        self.layers.append(initial_layer)
        self.params.extend(initial_layer.params)
        for _ in xrange(1,num_layers):
            inner_layer = NWLSTM_Layer(layer_num=_+1, word_dim=hidden_dim, 
                hidden_dim=hidden_dim, minibatch_dim=minibatch_dim, 
                activation=self.activation, want_stack=want_stack,
                stack_height=stack_height, push_vec=push_vec, pop_vec=pop_vec)
            self.layers.append(inner_layer)
            self.params.extend(inner_layer.params)
        
        W_hy = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (word_dim, hidden_dim))
        self.W_hy = theano.shared(name='W_hy', value=W_hy.astype(theano.config.floatX))
        self.params.append(self.W_hy)
        #########################################################
        #########################################################
        # Create shared variables for optimization updates
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)
        #########################################################
        #########################################################
        # Create push/pop vector symbolic variables if flagged
        if self.want_stack:
            self.PUSH = T.addbroadcast(theano.shared(name='PUSH', value=push_vec.astype(theano.config.floatX)), 1)

            self.POP = T.addbroadcast(theano.shared(name='POP', value=pop_vec.astype(theano.config.floatX)), 1)
        #########################################################
        #########################################################
        # Symbolic input/output variables and test values (for when theano.config.compute_test_value='raise')
        x = T.tensor3('x', dtype=theano.config.floatX)
        y = T.tensor3('y', dtype=theano.config.floatX)
        learning_rate = T.scalar('learning_rate', dtype=theano.config.floatX)
        
        sequence_length = 7
        x.tag.test_value = np.random.uniform(0, 1, size=(sequence_length,self.word_dim,self.minibatch_dim)).astype(theano.config.floatX)
        y.tag.test_value = np.random.uniform(0, 1, size=(sequence_length,self.word_dim,self.minibatch_dim)).astype(theano.config.floatX)
        learning_rate.tag.test_value = .001
        #########################################################
        #########################################################
        # Scan functions
        def make_masks(x):
            nonsymbolic_masks = []
            for layer in self.layers:
                rng = T.shared_randomstreams.RandomStreams(np.random.randint(999999))
                mask = rng.binomial(p=1-self.dropout, size=(layer.hidden_dim,layer.minibatch_dim), dtype=theano.config.floatX)

                nonsymbolic_masks.append(mask)

            masks = T.stack(nonsymbolic_masks, axis=2)
            return masks

        def map_push_pop(x, push_vec, pop_vec):
            BroadcastToAllHiddenDims_Type = T.TensorType(theano.config.floatX, (True,True,True))
            is_push = BroadcastToAllHiddenDims_Type()
            is_pop = BroadcastToAllHiddenDims_Type()
            # Map input to {push,pop,internal}
            argm_xt = T.argmax(x, axis=0)
            argm_push = T.argmax(push_vec, axis=0)
            argm_pop = T.argmax(pop_vec, axis=0)
            is_push = T.eq(argm_xt, argm_push)
            is_pop = T.eq(argm_xt, argm_pop)

            is_push = is_push.reshape((self.minibatch_dim,1,1))
            is_pop = is_pop.reshape((self.minibatch_dim,1,1))

            return is_push, is_pop

        def forward_prop_step(x_t, masks):
            is_push, is_pop = map_push_pop(x_t, self.PUSH, self.POP)

            layer_input = x_t
            for i,layer in enumerate(self.layers):
                layer_input = layer.forward_prop(layer_input, is_push, is_pop)
                layer_input = layer_input*masks[:,:,i] # dropout
            
            o_t = self.W_hy.dot(layer_input)
            
            return o_t
        #########################################################
        #########################################################
        # Scan calls
        masks, _ = theano.scan(
            make_masks,
            sequences=x,
            outputs_info=[None])

        o, _ = theano.scan(
            forward_prop_step,
            sequences=[x,masks],
            outputs_info=[None],
            truncate_gradient=self.bptt_truncate)
        #########################################################
        #########################################################
        # Error calculation and model interface functions
        # comput softmax in numerically stable, column-wise way
        # move distribution axis to the end, collapse sequences/minibatches along first axis, calculate softmax
        # for each of the sequence_length*num_minibatches rows, then re-roll and swap axes back
        swapped_o = T.swapaxes(o,1,2)
        swapped_flat_o = swapped_o.reshape((-1,swapped_o.shape[-1]))
        swapped_flat_o = np.exp(swapped_flat_o/self.softmax_temperature)
        swapped_flat_o = swapped_flat_o / swapped_flat_o.sum(axis=1, keepdims=True)
        swapped_o = swapped_flat_o.reshape(swapped_o.shape)
        softmaxed_o = T.swapaxes(swapped_o,1,2)

        # clip softmaxed probabilites to avoid explosion/vanishing during crossentropy calculation
        clipped_softmaxed_o = T.clip(softmaxed_o, 1e-6, 1 - 1e-6)
        #o_error = T.sum(T.nnet.categorical_crossentropy(clipped_softmaxed_o[-1,:,:], y[-1]))
        o_error = T.sum(T.nnet.categorical_crossentropy(clipped_softmaxed_o,y))

        self.forward_propagation = theano.function([x], softmaxed_o)
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

        def RMSprop(cost, params, learning_rate=0.01, b1=0.9, b2=0.999, epsilon=1e-6):
            grads = T.grad(cost=cost, wrt=params, consider_constant=[masks])
            updates = []
            for p, g in zip(params, grads):
                m_old = theano.shared(p.get_value() * 0.)
                v_old = theano.shared(p.get_value() * 0.)

                m = b1*m_old + (1-b1)*g
                v = b2*v_old + (1-b2)*(g**2)

                update = (learning_rate*m) / (T.sqrt(v) + epsilon)
                
                # For bias parameters, set update to be broadcastable along minibatch dimension
                if p.broadcastable[1]:
                    update = T.addbroadcast(update, 1)

                updates.append((m_old, m))
                updates.append((v_old, v))
                updates.append((p, p-update))

            return updates

        if self.optimization=='RMSprop': updates = RMSprop(o_error, self.params, learning_rate=learning_rate)
        elif self.optimization=='SGD': updates = SGD(o_error, self.params, learning_rate=learning_rate)

        self.train_model = theano.function(inputs=[x,y,learning_rate],
                                      outputs=[],
                                      updates=updates)#, mode=NanGuardMode(nan_is_error=True,inf_is_error=True,big_is_error=True))
                                      
        ###########################################################################
        ###########################################################################

    def loss_for_minibatch(self, X, Y):
        return self.ce_error(X,Y)/1.
