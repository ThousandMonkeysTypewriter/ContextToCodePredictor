"""
addition.py

Core task-specific model definition file. Sets up encoder model, program embeddings, argument
handling.
"""
from tasks.env.config import CONFIG
import tensorflow as tf
import tflearn
import numpy as np


class AdditionCore():
    def __init__(self, hidden_dim=100, state_dim=128, batch_size=1):
        """
        Instantiate an Addition Core object, with the necessary hyperparameters.
        """
        self.hidden_dim, self.state_dim, self.bsz = hidden_dim, state_dim, batch_size
        self.env_dim = CONFIG["ENVIRONMENT_ROW"] * CONFIG["ENVIRONMENT_DEPTH"]  # 4 * 10 = 40
        self.arg_dim = CONFIG["ARGUMENT_NUM"] * CONFIG["ARGUMENT_DEPTH"]        # 3 * 10 = 30
        self.program_dim = CONFIG["PROGRAM_EMBEDDING_SIZE"]

        # Setup Environment Input Layer
        self.env_in = tf.placeholder(tf.float32, shape=[self.bsz, self.env_dim], name="Env_Input")

        # Setup Argument Input Layer
        self.arg_in = tf.placeholder(tf.float32, shape=[self.bsz, self.arg_dim], name="Arg_Input")

        # Setup Program ID Input Layer
        self.prg_in = tf.placeholder(tf.int32, shape=[self.bsz, 1], name='Program_ID')

        # Build Environment Encoder Network (f_enc)
        self.state_encoding = self.build_encoder()

        # Build Program Matrices
        self.program_key = tflearn.variable(name='Program_Keys', shape=[CONFIG["PROGRAM_NUM"],
                                                                        CONFIG["PROGRAM_KEY_SIZE"]],
                                            initializer='truncated_normal')
        self.program_embedding = self.build_program_store()

    def build_encoder(self):
        """
        Build the Encoder Network (f_enc) taking the environment state (env_in) and the program
        arguments (arg_in), feeding through a Multilayer Perceptron, to generate the state encoding
        (s_t).

        Reed, de Freitas only specify that the f_enc is a Multilayer Perceptron => As such we use
        two ELU Layers, up-sampling to a state vector with dimension 128.

        Reference: Reed, de Freitas [9]
        """
        merge = tflearn.merge([self.env_in, self.arg_in], 'concat')
        elu = tflearn.fully_connected(merge, self.hidden_dim, activation='elu')
        elu = tflearn.fully_connected(elu, self.hidden_dim, activation='elu')
        out = tflearn.fully_connected(elu, self.state_dim)
        return out

    def get_incoming_shape(self, incoming):
        """ Returns the incoming data shape """
        if isinstance(incoming, tf.Tensor):
            return incoming.get_shape().as_list()
        elif type(incoming) in [np.array, list, tuple]:
            return np.shape(incoming)
        else:
            raise Exception("Invalid incoming layer.")

    def embedding(self, incoming, input_dim, output_dim, weights_init='truncated_normal',
                  trainable=True, restore=True, name="Embedding"):
        """ Embedding.

        Embedding layer for a sequence of ids.

        Input:
            2-D Tensor [samples, ids].

        Output:
            3-D Tensor [samples, embedded_ids, features].

        Arguments:
            incoming: Incoming 2-D Tensor.
            input_dim: list of `int`. Vocabulary size (number of ids).
            output_dim: list of `int`. Embedding size.
            weights_init: `str` (name) or `Tensor`. Weights initialization.
                (see tflearn.initializations) Default: 'truncated_normal'.
            trainable: `bool`. If True, weights will be trainable.
            restore: `bool`. If True, this layer weights will be restored when
                loading a model
            name: A name for this layer (optional). Default: 'Embedding'.

        """

        input_shape = self.get_incoming_shape(incoming)
        assert len(input_shape) == 2, "Incoming Tensor shape must be 2-D"
        n_inputs = int(np.prod(input_shape[1:]))

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = tflearn.initializations.get(weights_init)()

        with tf.name_scope(name) as scope:
            with tf.device('/cpu:0'):
                W = tf.get_variable(scope + "W", shape=[input_dim, output_dim], dtype=tf.float32,
                                      initializer=W_init,
                                      trainable=trainable)
                tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, W)

            inference = tf.cast(incoming, tf.int32)
            inference = tf.nn.embedding_lookup(W, inference)
            inference = tf.transpose(inference, [1, 0, 2])
            inference = tf.reshape(inference, shape=[-1, output_dim])
            inference = tf.split(inference, n_inputs)

        # TODO: easy access those var
        # inference.W = W
        # inference.scope = scope

        return inference

    def build_program_store(self):
        """
        Build the Program Embedding (M_prog) that takes in a specific Program ID (prg_in), and
        returns the respective Program Embedding.

        Reference: Reed, de Freitas [4]
        """
        print(self.prg_in, CONFIG["PROGRAM_NUM"],
                                      CONFIG["PROGRAM_EMBEDDING_SIZE"], "Program_Embedding")
        embedding = self.embedding(self.prg_in, CONFIG["PROGRAM_NUM"],
                                      CONFIG["PROGRAM_EMBEDDING_SIZE"], name="Program_Embedding")
        return embedding
