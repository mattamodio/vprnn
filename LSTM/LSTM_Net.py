import sys, operator
import numpy as np
import theano
from theano import tensor as T, function, printing, gradient
from utils import *
from LSTM_Layer import LSTM_Layer
from theano.compile.nanguardmode import NanGuardMode

class LSTM_Net:
    def __init__(self, word_dim, hidden_dim=100, minibatch_dim=1, bptt_truncate=4, num_layers=2,
        optimization='RMSprop', activation='tanh', softmax_temp=1, null_vec=None, dropout=1, l1_rate=.01, l2_rate=.01):
        #########################################################
        #########################################################
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.minibatch_dim = minibatch_dim
        self.bptt_truncate = bptt_truncate
        self.optimization = optimization
        #self.softmax_temperature = softmax_temperature
        self.dropout = dropout
        self.num_layers = num_layers
        self.l1_rate = l1_rate
        self.l2_rate = l2_rate
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
        initial_layer = LSTM_Layer(layer_num=1, word_dim=word_dim,hidden_dim=hidden_dim, minibatch_dim=minibatch_dim, activation=self.activation)
        self.layers.append(initial_layer)
        self.params.extend(initial_layer.params)

        for _ in xrange(1,num_layers):
            inner_layer = LSTM_Layer(layer_num=_+1, word_dim=hidden_dim, hidden_dim=hidden_dim, minibatch_dim=minibatch_dim, activation=self.activation)
            self.layers.append(inner_layer)
            self.params.extend(inner_layer.params)
        
        W_hy = np.random.uniform(-.01, .01, (word_dim, hidden_dim))
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
        # Helper functions
        def initialize():
            x = T.tensor3('x', dtype=theano.config.floatX)
            y = T.tensor3('y', dtype=theano.config.floatX)
            softmax_temperature = T.scalar('softmax_temperature', dtype=theano.config.floatX)
            learning_rate = T.scalar('learning_rate', dtype=theano.config.floatX)
            h_init = T.tensor3('h_init', dtype=theano.config.floatX)
            c_init = T.tensor3('c_init', dtype=theano.config.floatX)

            
            sequence_length = self.minibatch_dim+1
            x.tag.test_value = np.random.uniform(0, 1, size=(sequence_length,self.word_dim,self.minibatch_dim)).astype(theano.config.floatX)
            y.tag.test_value = np.random.uniform(0, 1, size=(sequence_length,self.word_dim,self.minibatch_dim)).astype(theano.config.floatX)
            learning_rate.tag.test_value = .001
            softmax_temperature.tag.test_value = 1
            h_init.tag.test_value = np.random.uniform(-.01,.01, (len(self.layers),self.hidden_dim,self.minibatch_dim)).astype(theano.config.floatX)
            c_init.tag.test_value = np.random.uniform(-.01,.01, (len(self.layers),self.hidden_dim,self.minibatch_dim)).astype(theano.config.floatX)

            return x,y,softmax_temperature,learning_rate,h_init,c_init

        def make_masks(x):
            nonsymbolic_masks = []
            for layer in self.layers:
                rng = T.shared_randomstreams.RandomStreams(np.random.randint(999999))
                mask = rng.binomial(p=self.dropout, size=(layer.hidden_dim,layer.minibatch_dim), dtype=theano.config.floatX)

                nonsymbolic_masks.append(mask)

            # T.stack gives (minibatch_dim,hidden_dim,layer_num) and we want (hidden_dim,minibatch_dim,layer_num) for point-wise multiplication
            masks = T.stack(nonsymbolic_masks)
            masks = T.swapaxes(masks, 0, 2)
            masks = T.swapaxes(masks, 0, 1)

            return masks

        def columnwise_softmax(o):
            # comput softmax in numerically stable, column-wise way
            # move distribution axis to the end, collapse sequences/minibatches along first axis, calculate softmax
            # for each of the sequence_length*num_minibatches rows, then re-roll and swap axes back
            swapped_o = T.swapaxes(o,1,2)
            swapped_flat_o = swapped_o.reshape((-1,swapped_o.shape[-1]))
            clipped_swapped_flat_o1 = T.clip(swapped_flat_o, -5., 5.) # don't exponentiate numbers too big/small
            clipped_swapped_flat_o2 = T.exp(clipped_swapped_flat_o1 / softmax_temperature)
            clipped_swapped_flat_o3 = clipped_swapped_flat_o2 / clipped_swapped_flat_o2.sum(axis=1, keepdims=True)
            softmaxed_swapped_o = clipped_swapped_flat_o3.reshape(swapped_o.shape)
            softmaxed_o = T.swapaxes(softmaxed_swapped_o,1,2)

            return softmaxed_o

        def forward_prop_step(x_t, masks, h_prevs, c_prevs):
            nonsymbolic_hs = []
            nonsymbolic_cs = []

            h = x_t
            for i,layer in enumerate(self.layers):
                h,c = layer.forward_prop(h, h_prevs[i,:,:], c_prevs[i,:,:])
                h = h*masks[:,:,i] / self.dropout # inverted dropout for scaling

                nonsymbolic_hs.append(h)
                nonsymbolic_cs.append(c)
            
            h_s = T.stack(nonsymbolic_hs)
            c_s = T.stack(nonsymbolic_cs)


            o_t = self.W_hy.dot(h)
            
            return o_t, h_s, c_s

        x,y,softmax_temperature,learning_rate,h_init,c_init = initialize()
        #########################################################
        #########################################################
        # Scan calls

        masks, _ = theano.scan(
            make_masks,
            sequences=x,
            outputs_info=[None])

        (o,h,c), _ = theano.scan(
            forward_prop_step,
            sequences=[x,masks],
            outputs_info=[None,h_init,c_init],
            truncate_gradient=self.bptt_truncate)
        #########################################################
        #########################################################
        # Error calculation and model interface functions
        softmaxed_o = columnwise_softmax(o)
        clipped_softmaxed_o = T.clip(softmaxed_o, 1e-6, 1 - 1e-6)

        o_error = T.sum( T.nnet.binary_crossentropy(clipped_softmaxed_o,y) )

        L1 = T.sum([abs(p).sum() for p in self.params if 'W' in p.name]) / T.cast(len([0 for p in self.params if 'W' in p.name]), theano.config.floatX)
        L2 = T.sum([(p**2).sum() for p in self.params if 'W' in p.name]) / T.cast(len([0 for p in self.params if 'W' in p.name]), theano.config.floatX)

        regularized_error = o_error + l1_rate*L1 + l2_rate*L2
        ###########################################################################
        ###########################################################################
        # Optimization methods
        def SGD(cost, params, learning_rate, disconnected_inputs='raise'):
            grads = T.grad(cost=cost, wrt=params, disconnected_inputs=disconnected_inputs)
            updates = []
            for p,g in zip(params, grads):
                update = (p, p - learning_rate*g)
                updates.append(update)
            return updates

        def RMSprop(cost, params, learning_rate=0.01, b1=0.9, b2=0.999, epsilon=1e-6, disconnected_inputs='raise'):
            grads = theano.grad(cost=cost, wrt=params, consider_constant=[masks], disconnected_inputs=disconnected_inputs)
            updates = []
            for p, g in zip(params, grads):          
                
                m_old = theano.shared(p.get_value() * 0.)
                v_old = theano.shared(p.get_value() * 0.)
                

                m = b1*m_old + (1-b1)*g
                v = b2*v_old + (1-b2)*(g**2)

                update = (learning_rate*m)# / (T.sqrt(v) + epsilon)
                
                # For bias parameters, set update to be broadcastable along minibatch dimension
                if p.broadcastable[1]:
                    update = T.addbroadcast(update, 1)

                updates.append((m_old, m))
                updates.append((v_old, v))
                updates.append((p, p-update))

            return updates

        if self.optimization=='RMSprop': updates = RMSprop(regularized_error, self.params, learning_rate=learning_rate)
        elif self.optimization=='SGD': updates = SGD(regularized_error, self.params, learning_rate=learning_rate)


        self.train_model = theano.function(inputs=[x,y,h_init,c_init,learning_rate,softmax_temperature],
                                      outputs=[h[self.minibatch_dim-1,:,:],c[self.minibatch_dim-1,:,:]],
                                      updates=updates)#, mode=NanGuardMode(nan_is_error=True,inf_is_error=True,big_is_error=True))

        self.ce_error = theano.function([x, y, h_init, c_init, softmax_temperature], [o_error, L1, L2])
        self.forward_propagation = theano.function([x, h_init, c_init, softmax_temperature], [softmaxed_o, h[-1,:,:], c[-1,:,:]])
        ###########################################################################
        ###########################################################################


    def build_pretrain(self):

        softmax_temperature = T.scalar('softmax_temperature', dtype=theano.config.floatX)
        learning_rate = T.scalar('learning_rate', dtype=theano.config.floatX)
        learning_rate.tag.test_value = .01
        softmax_temperature.tag.test_value = 1

        pretrain_Why = np.random.uniform(-.01, .01, (self.word_dim, self.hidden_dim))
        self.pretrain_Why = theano.shared(name='pretrain_Why', value=pretrain_Why.astype(theano.config.floatX))
        
        pretrain_h_init = np.random.uniform(-.01,.01, (len(self.layers),self.hidden_dim,self.minibatch_dim)).astype(theano.config.floatX)
        pretrain_c_init = np.random.uniform(-.01,.01, (len(self.layers),self.hidden_dim,self.minibatch_dim)).astype(theano.config.floatX)
        self.pretrain_h_init = theano.shared(name='pretrain_h_init', value=pretrain_h_init.astype(theano.config.floatX))
        self.pretrain_c_init = theano.shared(name='pretrain_c_init', value=pretrain_c_init.astype(theano.config.floatX))

        pretrain_x = T.tensor3('pretrain_x', dtype=theano.config.floatX)
        pretrain_x.tag.test_value = np.random.uniform(0, 1, size=(1,self.word_dim,self.minibatch_dim)).astype(theano.config.floatX)

        def columnwise_softmax(o):
            # comput softmax in numerically stable, column-wise way
            # move distribution axis to the end, collapse sequences/minibatches along first axis, calculate softmax
            # for each of the sequence_length*num_minibatches rows, then re-roll and swap axes back
            swapped_o = T.swapaxes(o,1,2)
            swapped_flat_o = swapped_o.reshape((-1,swapped_o.shape[-1]))
            clipped_swapped_flat_o1 = T.clip(swapped_flat_o, -5., 5.) # don't exponentiate numbers too big/small
            clipped_swapped_flat_o2 = T.exp(clipped_swapped_flat_o1 / softmax_temperature)
            clipped_swapped_flat_o3 = clipped_swapped_flat_o2 / clipped_swapped_flat_o2.sum(axis=1, keepdims=True)
            softmaxed_swapped_o = clipped_swapped_flat_o3.reshape(swapped_o.shape)
            softmaxed_o = T.swapaxes(softmaxed_swapped_o,1,2)

            return softmaxed_o

        def RMSprop(cost, params, learning_rate=0.01, b1=0.9, b2=0.999, epsilon=1e-6, disconnected_inputs='raise'):
            grads = theano.grad(cost=cost, wrt=params, disconnected_inputs=disconnected_inputs)
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

        def SGD(cost, params, learning_rate, disconnected_inputs='raise'):
            grads = T.grad(cost=cost, wrt=params, disconnected_inputs=disconnected_inputs)
            updates = []
            for p,g in zip(params, grads):
                update = (p, p - learning_rate*g)
                updates.append(update)
            return updates

        def pretrain_forward_prop(x_t, h_prevs, c_prevs):

            nonsymbolic_hs = []
            nonsymbolic_cs = []

            h = x_t
            for i,layer in enumerate(self.layers):
                h,c = layer.forward_prop(h, h_prevs[i,:,:], c_prevs[i,:,:])

                nonsymbolic_hs.append(h)
                nonsymbolic_cs.append(c)
            
            h_s = T.stack(nonsymbolic_hs)
            c_s = T.stack(nonsymbolic_cs)


            o_t = self.pretrain_Why.dot(h)
            
            return o_t, h_s, c_s

        (pretrain_o,pretrain_h,pretrain_c), _ = theano.scan(
            pretrain_forward_prop,
            sequences=[pretrain_x],
            outputs_info=[None,self.pretrain_h_init,self.pretrain_c_init],
            truncate_gradient=self.bptt_truncate)

        pretrain_softmaxed_o = columnwise_softmax(pretrain_o)
        pretrain_clipped_softmaxed_o = T.clip(pretrain_softmaxed_o, 1e-6, 1 - 1e-6)

        L2 = T.sum([(p**2).sum() for p in self.params if 'W' in p.name]) / T.cast(len([0 for p in self.params if 'W' in p.name]), theano.config.floatX)

        pretrain_error = T.sum( T.nnet.categorical_crossentropy(pretrain_clipped_softmaxed_o,pretrain_x) )

        pretrain_params = [p for p in self.params if p.name!='W_hy'] + [self.pretrain_Why,self.pretrain_h_init,self.pretrain_c_init]


        pretrain_updates = RMSprop(pretrain_error, pretrain_params, learning_rate=learning_rate, disconnected_inputs='ignore')

        self.pretrain_model = theano.function(inputs=[pretrain_x, learning_rate, softmax_temperature],
                                              outputs=[pretrain_clipped_softmaxed_o, self.pretrain_h_init, self.pretrain_c_init],
                                              updates=pretrain_updates, on_unused_input='warn')



    def loss_for_minibatch(self, X, Y, h, c, softmax):

        return [float(loss_component) for loss_component in self.ce_error(X, Y, h, c, softmax)]

    def loss_for_minibatch_stack(self, X, Y, h, c, stack, ptrs, softmax):

        return [float(loss_component) for loss_component in self.ce_error_stack(X, Y, h, c, stack, ptrs, softmax)]

    def getCode(self, char):
        if char in self.char_to_code_dict:
            code = self.char_to_code_dict[char]
        else:
            code = self.char_to_code_dict['NULL']
        return code





