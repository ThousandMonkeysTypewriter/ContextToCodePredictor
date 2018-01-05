"""
npi.py
Core model definition script for the Neural Programmer-Interpreter.
"""
import tensorflow as tf
from tensorflow.python.ops import init_ops
import tflearn
import numpy as np
from tasks.env.config import get_incoming_shape

def _linear(args, output_size, bias, W=None, b=None, W_init=None,
           bias_start=0.0, trainable=True, restore=True, scope=None):
    """ Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Arguments:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: `int`. Second dimension of W[i].
        bias: `bool`. Whether to add a bias term or not.
        W: `Tensor`. The weights. If None, it will be automatically created.
        b: `Tensor`. The bias. If None, it will be automatically created.
        W_init: `str`. Weights initialization mode. See
            tflearn.initializations.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
        A `tuple` containing:
        - W: `Tensor` variable holding the weights.
        - b: `Tensor` variable holding the bias.
        - res: `2D tf.Tensor` with shape [batch x output_size] equal to
            sum_i(args[i] * W[i]).

    """
    # Creates W if it hasn't be created yet.
    if not W:
        assert args
        if not isinstance(args, (list, tuple)):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError(
                    "Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError(
                    "Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]
        with tf.variable_scope(scope, reuse=False):
            W = tf.get_variable(name="W", shape=[total_arg_size, output_size],
                                initializer=W_init, trainable=trainable)
            if not restore:
                tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS, W)

    # Now the computation.
    if len(args) == 1:
        res = tf.matmul(args[0], W)
    else:
        res = tf.matmul(tf.concat(args, 1), W)
    if not bias:
        return W, None, res

    # Creates b if it hasn't be created yet.
    if not b:
        with tf.variable_scope(scope, reuse=False) as vs:
            b = tf.get_variable(
                "b", [output_size],
                initializer=init_ops.constant_initializer(bias_start),
                trainable=trainable)
            if not restore:
                tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS, b)
    return W, b, res + b

class RNNCell(object):
    """ RNNCell.

    Abstract object representing an RNN cell.

    An RNN cell, in the most abstract setting, is anything that has
    a state -- a vector of floats of size self.state_size -- and performs some
    operation that takes inputs of size self.input_size. This operation
    results in an output of size self.output_size and a new state.

    """

    def __call__(self, inputs, state, scope):
        """ Run this RNN cell on inputs, starting from the given state.

        Arguments:
            inputs: 2D Tensor with shape [batch_size x self.input_size].
            state: 2D Tensor with shape [batch_size x self.state_size].
            scope: VariableScope for the created subgraph; defaults to
                class name.

        Returns:
            A pair containing:
            - Output: A 2D Tensor with shape [batch_size x self.output_size]
            - New state: A 2D Tensor with shape [batch_size x self.state_size].
        """
        raise NotImplementedError("Abstract method")

    @property
    def input_size(self):
        """Integer: size of inputs accepted by this cell."""
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """Integer: size of state used by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        """Return state tensor (shape [batch_size x state_size]) filled with 0.

        Arguments:
            batch_size: int, float, or unit Tensor representing the batch size.
            dtype: the data type to use for the state.

        Returns:
            A 2D Tensor of shape [batch_size x state_size] filled with zeros.
        """
        zeros = tf.zeros(
            tf.pack([batch_size, self.state_size]), dtype=dtype)
        zeros.set_shape([None, self.state_size])
        return zeros

class BasicLSTMCell(RNNCell):
    """ Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/pdf/1409.2329v5.pdf.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    Biases of the forget gate are initialized by default to 1 in order to reduce
    the scale of forgetting in the beginning of the training.
    """

    def __init__(self, num_units, activation='sigmoid',
                 inner_activation='tanh', bias=True, W_init=None,
                 forget_bias=1.0, trainable=True, restore=True):
        self._num_units = num_units
        self._forget_bias = forget_bias
        if isinstance(activation, str):
            self.activation = tflearn.activations.get(activation)
        elif hasattr(activation, '__call__'):
            self.activation = activation
        else:
            raise ValueError("Invalid Activation.")
        if isinstance(inner_activation, str):
            self.inner_activation = tflearn.activations.get(inner_activation)
        elif hasattr(inner_activation, '__call__'):
            self.inner_activation = inner_activation
        else:
            raise ValueError("Invalid Activation.")
        self.W = None
        self.b = None
        if isinstance(W_init, str):
            W_init = initializations.get(W_init)()
        self.W_init = W_init
        self.bias = bias
        self.trainable = trainable
        self.restore = restore

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return 2 * self._num_units

    def __call__(self, inputs, state, scope):
        # Parameters of gates are concatenated into one multiply for efficiency.
        c, h = tf.split(state, 2, 1)
        self.W, self.b, concat = _linear([inputs, h], 4 * self._num_units,
                                         self.bias, self.W, self.b,
                                         self.W_init,
                                         trainable=self.trainable,
                                         restore=self.restore,
                                         scope=scope)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(concat, 4, 1)

        new_c = c * self.activation(f + self._forget_bias) + self.activation(
            i) * self.inner_activation(j)
        new_h = self.inner_activation(new_c) * self.activation(o)
        return new_h, tf.concat([new_c, new_h], 1)

class NPI():
    def __init__(self, core, config, log_path, npi_core_dim=256, npi_core_layers=2, verbose=0):
        """
        Instantiate an NPI Model, with the necessary hyperparameters, including the task-specific
        core.

        :param core: Task-Specific Core, with fields representing the environment state vector,
                     the input placeholders, and the program embedding.
        :param config: Task-Specific Configuration Dictionary, with fields representing the
                       necessary parameters.
        :param log_path: Path to save network checkpoint and tensorboard log files.
        """
        self.core, self.state_dim, self.program_dim = core, core.state_dim, core.program_dim
        self.bsz, self.npi_core_dim, self.npi_core_layers = core.bsz, npi_core_dim, npi_core_layers
        self.env_in, self.arg_in, self.prg_in = core.env_in, core.arg_in, core.prg_in
        self.state_encoding, self.program_embedding = core.state_encoding, core.program_embedding
        self.num_args, self.arg_depth = config["ARGUMENT_NUM"], config["ARGUMENT_DEPTH"]
        self.num_progs, self.key_dim = config["PROGRAM_NUM"], config["PROGRAM_KEY_SIZE"]
        self.log_path, self.verbose = log_path, verbose

        # Setup Label Placeholders
        self.y_term = tf.placeholder(tf.int64, shape=[None], name='Termination_Y')
        self.y_prog = tf.placeholder(tf.int64, shape=[None], name='Program_Y')
        self.y_args = [tf.placeholder(tf.int64, shape=[None, self.arg_depth],
                                      name='Arg{}_Y'.format(str(i))) for i in range(self.num_args)]

        # Build NPI LSTM Core, hidden state
        self.reset_state()
        self.h = self.npi_core()

        # Build Termination Network => Returns probability of terminating
        self.terminate = self.terminate_net()

        # Build Key Network => Generates probability distribution over programs
        self.program_distribution = self.key_net()

        # Build Argument Networks => Generates list of argument distributions
        self.arguments = self.argument_net()

        # Build Losses
        self.t_loss, self.p_loss, self.a_losses = self.build_losses()
        self.default_loss = 2 * self.t_loss + self.p_loss
        self.arg_loss = 0.25 * sum([self.t_loss, self.p_loss]) + sum(self.a_losses)

        # Build Optimizer
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.0001, self.global_step, 10000, 0.95,
                                                        staircase=True)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Build Metrics
        self.t_metric, self.p_metric, self.a_metrics = self.build_metrics()
        self.metrics = [self.t_metric, self.p_metric] + self.a_metrics

        # Build Train Ops
        self.default_train_op = self.opt.minimize(self.default_loss, global_step=self.global_step)
        self.arg_train_op = self.opt.minimize(self.arg_loss, global_step=self.global_step)

    def reset_state(self):
        """
        Zero NPI Core LSTM Hidden States. LSTM States are represented as a Tuple, consisting of the
        LSTM C State, and the LSTM H State (in that order: (c, h)).
        """
        zero_state = tf.zeros([self.bsz, 2 * self.npi_core_dim])
        self.h_states = [zero_state for _ in range(self.npi_core_layers)]

    def _rnn(self, cell, inputs, initial_state, dtype, sequence_length,
             scope_):
        """ Creates a recurrent neural network specified by RNNCell "cell".

        The simplest form of RNN network generated is:
          state = cell.zero_state(...)
          outputs = []
          states = []
          for input_ in inputs:
            output, state = cell(input_, state)
            outputs.append(output)
            states.append(state)
          return (outputs, states)

        However, a few other options are available:

        An initial state can be provided.
        If sequence_length is provided, dynamic calculation is performed.

        Dynamic calculation returns, at time t:
          (t >= max(sequence_length)
              ? (zeros(output_shape), zeros(state_shape))
              : cell(input, state)

        Thus saving computational time when unrolling past the max sequence length.

        Arguments:
          cell: An instance of RNNCell.
          inputs: A length T list of inputs, each a tensor of shape
            [batch_size, cell.input_size].
          initial_state: (optional) An initial state for the RNN.  This must be
            a tensor of appropriate type and shape [batch_size x cell.state_size].
          dtype: (optional) The data type for the initial state.  Required if
            initial_state is not provided.
          sequence_length: An int64 vector (tensor) size [batch_size].
          scope: VariableScope for the created subgraph; defaults to "RNN".

        Returns:
          A pair (outputs, states) where:
            outputs is a length T list of outputs (one for each input)
            states is a length T list of states (one state following each input)

        Raises:
          TypeError: If "cell" is not an instance of RNNCell.
          ValueError: If inputs is None or an empty list.
        """
        scope = scope_[:-1]
        if not isinstance(cell, RNNCell):
            raise TypeError("cell must be an instance of RNNCell")
        if not isinstance(inputs, list):
            raise TypeError("inputs must be a list")
        if not inputs:
            raise ValueError("inputs must not be empty")

        outputs = []
        states = []
        batch_size = tf.shape(inputs[0])[0]
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, dtype must be.")
            state = cell.zero_state(batch_size, dtype)

        if sequence_length:  # Prepare variables
            zero_output_state = (
                tf.zeros(tf.pack([batch_size, cell.output_size]),
                                inputs[0].dtype),
                tf.zeros(tf.pack([batch_size, cell.state_size]),
                                state.dtype))
            max_sequence_length = tf.reduce_max(sequence_length)

        for time, input_ in enumerate(inputs):
            def output_state():
                return cell(input_, state, scope)

            if sequence_length:
                (output, state) = control_flow_ops.cond(
                    time >= max_sequence_length,
                    lambda: zero_output_state, output_state)
            else:
                (output, state) = output_state()

            outputs.append(output)
            states.append(state)

        return (outputs, states)

    def lstm(self, incoming, n_units, activation='sigmoid', inner_activation='tanh',
             bias=True, weights_init='truncated_normal', forget_bias=1.0,
             return_seq=False, return_states=False, initial_state=None,
             sequence_length=None, trainable=True, restore=True, name="LSTM"):
        """ LSTM.

        Long Short Term Memory Recurrent Layer.

        Input:
            3-D Tensor [samples, timesteps, input dim].

        Output:
            if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
            else: 2-D Tensor [samples, output dim].

        Arguments:
            incoming: `Tensor`. Incoming 3-D Tensor.
            n_units: `int`, number of units for this layer.
            activation: `str` (name) or `function` (returning a `Tensor`).
                Activation applied to this layer (see tflearn.activations).
                Default: 'sigmoid'.
            inner_activation: `str` (name) or `function` (returning a `Tensor`).
                LSTM inner activation. Default: 'tanh'.
            bias: `bool`. If True, a bias is used.
            weights_init: `str` (name) or `Tensor`. Weights initialization.
                (See tflearn.initializations) Default: 'truncated_normal'.
            forget_bias: `float`. Bias of the forget gate. Default: 1.0.
            return_seq: `bool`. If True, returns the full sequence instead of
                last sequence output only.
            return_states: `bool`. If True, returns a tuple with output and
                states: (output, states).
            initial_state: `Tensor`. An initial state for the RNN.  This must be
                a tensor of appropriate type and shape [batch_size x cell.state_size].
            sequence_length: Specifies the length of each sequence in inputs.
                An int32 or int64 vector (tensor) size `[batch_size]`.
            trainable: `bool`. If True, weights will be trainable.
            restore: `bool`. If True, this layer weights will be restored when
                loading a model.
            name: `str`. A name for this layer (optional).

        References:
            Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber,
            Neural Computation 9(8): 1735-1780, 1997.

        Links:
            [http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf]
            (http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)

        """
        input_shape = self.get_incoming_shape(incoming)
        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = tflearn.initializations.get(weights_init)()

        with tf.name_scope(name) as scope:
            cell = BasicLSTMCell(n_units, activation, inner_activation, bias,
                                 W_init, forget_bias, trainable, restore)
            inference = incoming
            # If a tensor given, convert it to a per timestep list
            if type(inference) not in [list, np.array]:
                ndim = len(input_shape)
                assert ndim >= 3, "Input dim should be at least 3."
                axes = [1, 0] + list(range(2, ndim))
                inference = tf.transpose(inference, (axes))
                inference = tf.unstack(inference)

            outputs, states = self._rnn(cell, inference, initial_state, tf.float32, sequence_length, scope)
            # Track per layer variables
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, cell.W)
            if bias:
                tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope,
                                     cell.b)
            # Track activations.
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, outputs[-1])

        o = outputs if return_seq else outputs[-1]
        s = states if return_seq else states[-1]

        return (o, s) if return_states else o

    def npi_core(self):
        """
        Build the NPI LSTM core, feeding the program embedding and state encoding to a multi-layered
        LSTM, returning the h-state of the final LSTM layer.

        References: Reed, de Freitas [2]
        """
        s_in = self.state_encoding                               # Shape: [bsz, state_dim]
        p_in = self.program_embedding                            # Shape: [bsz, 1, program_dim]

        # Reshape state_in
        s_in = tflearn.reshape(s_in, [-1, 1, self.state_dim])    # Shape: [bsz, 1, state_dim]

        # Concatenate s_in, p_in
        c = tflearn.merge([s_in, p_in], 'concat', axis=2)        # Shape: [bsz, 1, state + prog]

        # Feed through Multi-Layer LSTM
        for i in range(self.npi_core_layers):
            c, [self.h_states[i]] = self.lstm(c, self.npi_core_dim, return_seq=True,
                                                 initial_state=self.h_states[i], return_states=True)

        # Return Top-Most LSTM H-State
        top_state = tf.split(self.h_states[-1], 2, 1)[1]
        return top_state                                         # Shape: [bsz, npi_core_dim]

    def terminate_net(self):
        """
        Build the NPI Termination Network, that takes in the NPI Core Hidden State, and returns
        the probability of terminating program.

        References: Reed, de Freitas [3]
        """
        p_terminate = tflearn.fully_connected(self.h, 2, activation='linear', regularizer='L2')
        return p_terminate                                      # Shape: [bsz, 2]

    def key_net(self):
        """
        Build the NPI Key Network, that takes in the NPI Core Hidden State, and returns a softmax
        distribution over possible next programs.

        References: Reed, de Freitas [3, 4]
        """
        # Get Key from Key Network
        hidden = tflearn.fully_connected(self.h, self.key_dim, activation='elu', regularizer='L2')
        key = tflearn.fully_connected(hidden, self.key_dim)    # Shape: [bsz, key_dim]

        # Perform dot product operation, then softmax over all options to generate distribution
        key = tflearn.reshape(key, [-1, 1, self.key_dim])
        key = tf.tile(key, [1, self.num_progs, 1])             # Shape: [bsz, n_progs, key_dim]
        prog_sim = tf.multiply(key, self.core.program_key)          # Shape: [bsz, n_progs, key_dim]
        prog_dist = tf.reduce_sum(prog_sim, [2])               # Shape: [bsz, n_progs]
        return prog_dist

    def argument_net(self):
        """
        Build the NPI Argument Networks (a separate net for each argument), each of which takes in
        the NPI Core Hidden State, and returns a softmax over the argument dimension.

        References: Reed, de Freitas [3]
        """
        args = []
        for i in range(self.num_args):
            arg = tflearn.fully_connected(self.h, self.arg_depth, activation='linear',
                                          regularizer='L2', name='Argument_{}'.format(str(i)))
            args.append(arg)
        return args                                             # Shape: [bsz, arg_depth]

    def build_losses(self):
        """
        Build separate loss computations, using the logits from each of the sub-networks.
        """
        # Termination Network Loss
        termination_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_term,
            logits=self.terminate), name='Termination_Network_Loss')

        # Program Network Loss
        program_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_prog,
            logits=self.program_distribution), name='Program_Network_Loss')

        # Argument Network Losses
        arg_losses = []
        for i in range(self.num_args):
            arg_losses.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_args[i],
                logits=self.arguments[i]), name='Argument{}_Network_Loss'.format(str(i))))

        return termination_loss, program_loss, arg_losses

    def build_metrics(self):
        """
        Build accuracy metrics for each of the sub-networks.
        """
        term_metric = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.terminate, 1),
                                                      self.y_term),
                                             tf.float32), name='Termination_Accuracy')

        program_metric = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.program_distribution, 1),
                                                         self.y_prog),
                                                tf.float32), name='Program_Accuracy')

        arg_metrics = []
        for i in range(self.num_args):
            arg_metrics.append(tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.arguments[i], 1), tf.argmax(self.y_args[i], 1)),
                        tf.float32), name='Argument{}_Accuracy'.format(str(i))))

        return term_metric, program_metric, arg_metrics
