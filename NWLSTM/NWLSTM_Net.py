import sys, operator
import numpy as np
import theano
from theano import tensor as T, function, printing, gradient
from utils import *
from NWLSTM_Layer import NWLSTM_Layer
from theano.compile.nanguardmode import NanGuardMode

class NWLSTM_Net:
    def __init__(self, word_dim, hidden_dim=100, minibatch_dim=1, bptt_truncate=4, num_layers=2,
        optimization='RMSprop', activation='tanh', want_stack=False, stack_height=None, 
        push_vec=None, pop_vec=None, softmax_temperature=1, dropout=0, l1_rate=.5, l2_rate=.5):
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
        elif activation=='relu': self.activation = T.nnet.relu #lambda x: x*(x>0)
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

        def make_h_c_inits():
            h_inits = []
            c_inits = []
            
            h_init = T.unbroadcast(T.zeros((self.hidden_dim, self.minibatch_dim)), 1)
            c_init = T.unbroadcast(T.zeros((self.hidden_dim, self.minibatch_dim)), 1)

            for _ in self.layers:
                h_inits.append(h_init)
                c_inits.append(c_init)

            return h_inits, c_inits

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

        def forward_prop_step(x_t, masks, stepcounter):

            # if it's the first element in the sequence, zero out the shared variables for hidden/cell states
            
            for i,layer in enumerate(self.layers):
                self.layers[i].h = T.switch(T.eq(stepcounter,0),
                    self.layers[i].h,
                    T.cast(T.unbroadcast(T.zeros((self.layers[i].h.shape[0], self.layers[i].h.shape[1])), 1), theano.config.floatX))
                self.layers[i].c = T.switch(T.eq(stepcounter,0),
                    self.layers[i].c,
                    T.cast(T.unbroadcast(T.zeros((self.layers[i].c.shape[0], self.layers[i].c.shape[1])), 1), theano.config.floatX))
                # theano.printing.Print('self.layers[i].c')(self.layers[i].c)
            # determine, for all layers, if this input was a push/pop
            is_push, is_pop = map_push_pop(x_t, self.PUSH, self.POP)

            layer_input = x_t
            for i,layer in enumerate(self.layers):
                layer_input = layer.forward_prop(layer_input, is_push, is_pop)
                layer_input = layer_input*masks[:,:,i] # dropout
            
            o_t = self.W_hy.dot(layer_input)
            
            return o_t
        
        def forward_prop_step2(x_t, masks, h_prevs, c_prevs):

            # determine, for all layers, if this input was a push/pop
            is_push, is_pop = map_push_pop(x_t, self.PUSH, self.POP)

            # theano.printing.Print('h_prevs')(h_prevs)
            # theano.printing.Print('h_prevs shape')(h_prevs.shape)
            # theano.printing.Print('h_prevs[0]')(h_prevs[0])
            # theano.printing.Print('h_prevs[0] shape')(h_prevs[0].shape)

            nonsymbolic_hs = []
            nonsymbolic_cs = []

            h,c = self.layers[0].forward_prop2(x_t, h_prevs[0], c_prevs[0], is_push, is_pop)
            h = h*masks[:,:,0] # dropout

            # theano.printing.Print('h')(h)
            # theano.printing.Print('h shape')(h.shape)

            nonsymbolic_hs.append(h)
            nonsymbolic_cs.append(c)

            for i,layer in enumerate(self.layers[1:]):
                h,c = layer.forward_prop2(h, h_prevs[i], c_prevs[i], is_push, is_pop)
                h = h*masks[:,:,i] # dropout
                # theano.printing.Print('h')(h)

                nonsymbolic_hs.append(h)
                nonsymbolic_cs.append(c)
            
            h_s = T.stack(nonsymbolic_hs, axis=0)
            c_s = T.stack(nonsymbolic_cs, axis=0)

            # theano.printing.Print('h_s shape')(h_s.shape)

            o_t = self.W_hy.dot(h)
            
            return o_t, h_s, c_s
        #########################################################
        #########################################################
        # Scan calls
        h_inits,c_inits = make_h_c_inits()

        masks, _ = theano.scan(
            make_masks,
            sequences=x,
            outputs_info=[None])

        (o,h,c), _ = theano.scan(
            forward_prop_step2,
            sequences=[x,masks],
            outputs_info=[None,h_inits,c_inits],
            truncate_gradient=self.bptt_truncate)

        # o, _ = theano.scan(
        #     forward_prop_step,
        #     sequences=[x,masks,T.arange(x.shape[0])],
        #     outputs_info=[None],
        #     truncate_gradient=self.bptt_truncate)
        #########################################################
        #########################################################
        # Error calculation and model interface functions
        # comput softmax in numerically stable, column-wise way
        # move distribution axis to the end, collapse sequences/minibatches along first axis, calculate softmax
        # for each of the sequence_length*num_minibatches rows, then re-roll and swap axes back
        # theano.printing.Print('o')(o)
        swapped_o = T.swapaxes(o,1,2)
        swapped_flat_o = swapped_o.reshape((-1,swapped_o.shape[-1]))
        clipped_swapped_flat_o1 = T.clip(swapped_flat_o, -5., 5.) # don't exponentiate numbers too big/small
        clipped_swapped_flat_o2 = np.exp(clipped_swapped_flat_o1 / self.softmax_temperature)
        clipped_swapped_flat_o3 = clipped_swapped_flat_o2 / clipped_swapped_flat_o2.sum(axis=1, keepdims=True)
        softmaxed_swapped_o = clipped_swapped_flat_o3.reshape(swapped_o.shape)
        softmaxed_o = T.swapaxes(softmaxed_swapped_o,1,2)
        # softmaxed_o = theano.printing.Print('softmaxed o')(softmaxed_o)
        

        # clip softmaxed probabilites to avoid explosion/vanishing during crossentropy calculation
        clipped_softmaxed_o = T.clip(softmaxed_o, 1e-6, 1 - 1e-6)
        o_error = T.sum(T.nnet.categorical_crossentropy(clipped_softmaxed_o,y)) / T.cast(x.shape[0], 'float32')
        #o_error = T.sum((o-y)**2)

        
        L1 = T.sum([abs(p).sum() for p in self.params if 'B' not in p.name])
        L2 = T.sum([(p**2).sum() for p in self.params if 'B' not in p.name])

        # theano.printing.Print('L1')(L1)
        # theano.printing.Print('L2')(L2)
        # theano.printing.Print('o_error')(o_error)

        regularized_error = o_error + l1_rate*L1 + l2_rate*L2
        self.ce_error = theano.function([x, y], o_error)
        self.forward_propagation = theano.function([x], softmaxed_o)
        #grads = theano.grad(cost=o_error, wrt=self.params, consider_constant=[masks])
        #self.get_grads = theano.function([x,y], grads)
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
            grads = theano.grad(cost=cost, wrt=params, consider_constant=[masks])
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

        if self.optimization=='RMSprop': updates = RMSprop(regularized_error, self.params, learning_rate=learning_rate)
        elif self.optimization=='SGD': updates = SGD(regularized_error, self.params, learning_rate=learning_rate)

        self.train_model = theano.function(inputs=[x,y,learning_rate],
                                      outputs=[],
                                      updates=updates)#, mode=NanGuardMode(nan_is_error=True,inf_is_error=True,big_is_error=True)                 
        ###########################################################################
        ###########################################################################

    def loss_for_minibatch(self, X, Y):
        return self.ce_error(X,Y)/1.



