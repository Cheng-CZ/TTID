import tensorflow as tf
import numpy as np
import collections
import scipy.sparse
from scipy.sparse import coo_matrix
from scipy.io import loadmat
from utils import show_all_variables

#Test for tf1.0
#from tf.contrib.rnn.core_rnn_cell import RNNCell
tfversion_ = tf.VERSION.split(".")
global tfversion
if int(tfversion_[0]) < 1:
    raise EnvironmentError("TF version should be above 1.0!!")

if int(tfversion_[1]) < 1:
    print("Working in TF version 1.0....")
    from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
    tfversion = "old"
else:
    print("Working in TF version 1.%d...." % int(tfversion_[1]))
    from tensorflow.python.ops.rnn_cell_impl import RNNCell
    tfversion = "new"


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ('c', 'h'))

class LSTMStateTuple(_LSTMStateTuple):
    __slots__ = () #What is this??
    
    @property
    def dtype(self):
        (c, h) = self
        if not c.dtype == h.dtype:
            raise TypeError("Inconsistent internal state")
        return c.dtype

class LSTMCell(RNNCell):
    def __init__(self, num_units, forget_bias=1.0,
                state_is_tuple=True, activation=None, reuse=None,
                feat_in=None, nNode=None):
        if tfversion == 'new':
            super(LSTMCell, self).__init__(_reuse=reuse) #super what is it?
        
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or tf.tanh
        self._feat_in = feat_in
        self._nNode = nNode
        
        
    @property
    def state_size(self):
        return(LSTMStateTuple((self._nNode, self._num_units), (self._nNode, self._num_units))
              if self._state_is_tuple else 2*self._num_units)
    @property
    def output_size(self):
        return self._num_units
    
    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "myZeroState"):
            # zero_state_c = tf.zeros([batch_size, self._nNode, self._num_units], name='c')
            zero_state_c = tf.zeros([self._feat_in, self._nNode], name='c')

            # zero_state_h = tf.zeros([batch_size, self._nNode, self._num_units], name='h')
            zero_state_h = tf.zeros([self._feat_in, self._nNode], name='h')

            #print("When it called, I print batch_size", batch_size)
            return (zero_state_c, zero_state_h)
    
    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)
            feat_in = self._feat_in
            nNode = self._nNode
        
            if feat_in is None:
                batch_size, nNode, feat_in = inputs.get_shape()
                print("hey!")
                
            dim_h = feat_in # 5 for 5 topics
            
            inputs = tf.transpose(tf.squeeze(inputs)) # feat_in * nNode
            
            scope = tf.get_variable_scope()
            with tf.variable_scope(scope) as scope:
                
                # x
                Wcxt = tf.get_variable("Wcxt", [dim_h, dim_h], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                Wixt = tf.get_variable("Wixt", [dim_h, dim_h], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                Wfxt = tf.get_variable("Wfxt", [dim_h, dim_h], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                Woxt = tf.get_variable("Woxt", [dim_h, dim_h], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                # h
                Wcht = tf.get_variable("Wcht", [dim_h, dim_h], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                Wiht = tf.get_variable("Wiht", [dim_h, dim_h], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                Wfht = tf.get_variable("Wfht", [dim_h, dim_h], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                Woht = tf.get_variable("Woht", [dim_h, dim_h], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                # c
                # Wcct = tf.get_variable("Wcct", [K*feat_in, dim_h], dtype=tf.float32,
                #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                Wict = tf.get_variable("Wict", [dim_h, dim_h], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                Wfct = tf.get_variable("Wfct", [dim_h, dim_h], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                Woct = tf.get_variable("Woct", [dim_h, dim_h], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))                    
                
                bct = tf.get_variable("bct", [dim_h, 1], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                bit = tf.get_variable("bit", [dim_h, 1], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                bft = tf.get_variable("bft", [dim_h, 1], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                bot = tf.get_variable("bot", [dim_h, 1], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                # LSTM operations
                cxt = tf.matmul(Wcxt, inputs)
                cht = tf.matmul(Wcht, h)
                ct  = cxt + cht + bct
                ct  = tf.tanh(ct)
                
                ixt = tf.matmul(Wixt, inputs)
                iht = tf.matmul(Wiht, h)
                ict = tf.matmul(Wict, c)
                it  = ixt + iht + ict + bit
                it  = tf.sigmoid(it)
                
                fxt = tf.matmul(Wfxt, inputs)
                fht = tf.matmul(Wfht, h)
                fct = tf.matmul(Wfct, c)
                ft  = fxt + fht + fct + bft
                ft  = tf.sigmoid(ft)
                
                oxt = tf.matmul(Woxt, inputs)
                oht = tf.matmul(Woht, h)
                oct_ = tf.matmul(Woct, c)
                ot  = oxt + oht + oct_ + bot
                ot  = tf.sigmoid(ot)
                
                # c
                new_c = ft*c + it*ct
                
                # h
                new_h = ot*tf.tanh(new_c)
                
                if self._state_is_tuple:
                    new_state = LSTMStateTuple(new_c, new_h)
                else:
                    new_state = tf.concat([new_c, new_h], 1)
                return new_h, new_state

class Model(object):
    """
    Defined:
        Placeholder
        Model architecture
        Train / Test function
    """
    
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.num_node = config.num_node
        self.feat_in = config.feat_in
        self.num_time_steps = config.num_time_steps
        self.feat_out = config.feat_out
        
        self.num_hidden = config.num_hidden
        self.learning_rate = config.learning_rate
        self.max_grad_norm = None
        if config.max_grad_norm > 0:
            self.max_grad_norm = config.max_grad_norm
        self.optimizer = config.optimizer
        
        self._build_placeholders()
        self._build_model()
        self._build_steps()
        self._build_optim()
        
        show_all_variables()

    def _build_placeholders(self):
        
        self.rnn_input = tf.placeholder(tf.float32, 
                                    [self.num_time_steps, self.num_node, self.feat_in],
                                    name="rnn_input")

        # 10 * 5 * 1031 * 1031
        self.rnn_adj = tf.placeholder(tf.float32, 
                                    [self.num_time_steps, 5, self.num_node, self.num_node],
                                    name="rnn_adj")

        self.model_step = tf.Variable(
            0, name='model_step', trainable=False)
            
    def _build_model(self, reuse=None):
        with tf.variable_scope("gconv_model", reuse=reuse) as sc:
            
            cell = LSTMCell(num_units=self.num_hidden, forget_bias=1.0, 
                             feat_in=self.feat_in, 
                             nNode=self.num_node)
            # output_variable = {
            #     'weight': tf.Variable(tf.random_normal([self.num_hidden, self.feat_out])),
            #     'bias' : tf.Variable(tf.random_normal([self.feat_out]))}

            # mention, quote, retweet, hashtag, follow
            self.adj_variable = tf.Variable(tf.random_normal([5, 1, 1], mean=1))

            adj_mulplied = tf.multiply(self.adj_variable, self.rnn_adj)
            
            # self.rnn_sum_adj = tf.reduce_sum(self.rnn_adj, axis=1)
            self.rnn_sum_adj = tf.reduce_sum(adj_mulplied, axis=1)


            self.rnn_input_seq = tf.matmul(self.rnn_sum_adj, self.rnn_input) + self.rnn_input

            #normalize adj matrix
            # self.rnn_input_seq = tf.nn.l2_normalize(self.rnn_input_seq, dim=[1,2])
            self.rnn_input_seq = self.rnn_input_seq / tf.norm(self.rnn_input_seq)
            self.rnn_input_seq = tf.unstack(self.rnn_input_seq, self.num_time_steps, 0)
                
            if tfversion == 'new':
                # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)
                outputs, states = tf.nn.static_rnn(cell, self.rnn_input_seq, dtype=tf.float32)
            else:
                outputs, states = tf.contrib.rnn.static_rnn(cell, self.rnn_input_seq, dtype=tf.float32)
            #cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)
            #Check the tf version here

            # for output_ in outputs:
            #     print output_.eval(session=sess)

            predictions = []
            for output in outputs:
                # output_reshape = tf.reshape(output, [-1, self.num_hidden])
                # prediction = tf.matmul(output_reshape, output_variable['weight']) + output_variable['bias']
                prediction = output
                prediction = tf.reshape(prediction, [1, self.feat_out, self.num_node])
                predictions.append(prediction)
            
            
            self.pred_out = tf.concat(predictions, 0)

            predictions = []
            for output in outputs:
                # output_reshape = tf.reshape(output, [-1, self.num_hidden])
                # prediction = tf.matmul(output_reshape, output_variable['weight']) + output_variable['bias']
                prediction = output
                prediction = tf.reshape(prediction, [self.feat_out, self.num_node])
                predictions.append(prediction)

            #pred_out_softmax = tf.nn.softmax(pred_out,dim=1)
            self.predictions = predictions
            self.model_vars = tf.contrib.framework.get_variables(
                sc, collection=tf.GraphKeys.TRAINABLE_VARIABLES)
            
        self._build_loss()
        
    def _build_loss(self):
            
        predictions = tf.stack(self.predictions, axis=0) # 10 * 5 * 1031
        Ht_ = predictions[1:]
        Ht_1 = predictions[:-1]
        H_d = Ht_ - Ht_1
        losses1 = tf.reduce_sum(tf.norm(H_d, axis=[1,2])) # first term

        x_inputs = tf.stack(self.rnn_input_seq, axis=0) # 10 * 1031 * 5
        Xt_ = x_inputs[1:]
        Xt_1 = x_inputs[:-1]
        X_d = Xt_ - Xt_1
        one_5_2 = tf.ones([9, 5, 1], tf.float32)
        T2_first = tf.matmul(tf.multiply(X_d, X_d), one_5_2)
        Ht_T = tf.transpose(Ht_, perm=[0, 2, 1])
        T2_second = tf.matmul(tf.multiply(Ht_T, Ht_T), one_5_2)
        losses2 = tf.matmul(tf.transpose(T2_first, perm=[0, 2, 1]), T2_second)
        losses2 = tf.reduce_sum(losses2)

        A_input = self.rnn_sum_adj # 10 * 1031 * 1031
        one_1031 = tf.ones([10, 1031, 1], tf.float32)
        T3_first = tf.matmul(A_input, (tf.matmul(A_input, one_1031) + one_1031))
        Ht = tf.transpose(predictions, perm=[0, 2, 1])
        one_5_3 = tf.ones([10, 5, 1], tf.float32)
        T3_second = tf.matmul(tf.multiply(Ht, Ht), one_5_3)
        losses3 = tf.matmul(tf.transpose(T3_first, perm=[0, 2, 1]), T3_second)
        losses3 = tf.reduce_mean(losses3)

        one_5_4 = tf.transpose(one_5_3, perm=[0, 2, 1])
        T4_first = tf.matmul(tf.transpose(one_1031, perm=[0, 2, 1]), Ht) - one_5_4
        T4_firstsecond = tf.multiply(T4_first, T4_first)
        T4_third = one_5_3
        losses4 = tf.reduce_sum(tf.matmul(T4_firstsecond, T4_third))


        losses = losses1 - losses3 - losses2 + losses4
            
        
        with tf.name_scope("losses"):
            self.loss = losses
            
        self.model_summary = tf.summary.merge([tf.summary.scalar("model_loss/influential_loss",
                                                           self.loss)])
        #if hasattr(self, "model_summary"):
        #    self.model_summary = loss_summary
            
    def _build_steps(self):
        def run(sess, feed_dict, fetch,
                summary_op, summary_writer, output_op=None, output_img=None):
            if summary_writer is not None:
                fetch['summary'] = summary_op
            if output_op is not None:
                fetch['output'] = output_op
                fetch['adj'] = self.adj_variable

            result = sess.run(fetch, feed_dict=feed_dict)
            if "summary" in result.keys() and "step" in result.keys():
                summary_writer.add_summary(result['summary'], result['step'])
                summary_writer.flush()
            return result
        
        def train(sess, feed_dict, summary_writer=None,
                  with_output=False):
            fetch = {'loss': self.loss,
                     'optim': self.model_optim, #?
                     'step': self.model_step
            }
            return run(sess, feed_dict, fetch,
                       self.model_summary, summary_writer,
                       output_op=self.pred_out if with_output else None,)
        
        def test(sess, feed_dict, summary_writer=None,
                 with_output=False):
            fetch = {'loss': self.loss,
                    'test_step': self.model_step}
            return run(sess, feed_dict, fetch,
                       self.model_summary, summary_writer,
                       output_op=self.pred_out if with_output else None,)
        self.train = train
        self.test = test
        
    def _build_optim(self):
        def minimize(loss, step, var_list, learning_rate, optimizer):
            if optimizer == "sgd":
                optim = tf.train.GradientDescentOptimizer(learning_rate)
            elif optimizer == "adam":
                optim = tf.train.AdamOptimizer(learning_rate)
            elif optimizer == "rmsprop":
                optim = tf.train.RMSPropOptimizer(learning_rate)
            else:
                raise Exception("[!] Unkown optimizer: {}".format(
                    optimizer))
            ## Gradient clipping ##    
            if self.max_grad_norm is not None:
                grads_and_vars = optim.compute_gradients(
                    loss, var_list=var_list)
                self.grad_var = grads_and_vars
                new_grads_and_vars = []
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None and var in var_list:
                        grad = tf.clip_by_norm(grad, self.max_grad_norm)
                        grad = tf.check_numerics(
                            grad, "Numerical error in gradient for {}".format(
                                var.name))
                        new_grads_and_vars.append((grad, var))
                return optim.apply_gradients(new_grads_and_vars, global_step=step)
            else:
                grads_and_vars = optim.compute_gradients(
                    loss, var_list=var_list)
                self.grad_var = grads_and_vars
                return optim.apply_gradients(grads_and_vars,
                                             global_step=step)
        
        # optim #
        self.model_optim = minimize(
            self.loss,
            self.model_step,
            self.model_vars,
            self.learning_rate,
            self.optimizer)
        
