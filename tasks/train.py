"""
train.py

Core training script for the addition task-specific NPI. Instantiates a model, then trains using
the precomputed data.
"""
from model.npi import NPI
from tasks.env.addition import AdditionCore
from dsl.dsl import DSL
from tasks.env.config import CONFIG, get_args, LOG_PATH, DATA_PATH_TRAIN, CKPT_PATH
import dsl.dsl
import pickle
import tensorflow as tf
import numpy as np


MOVE_PID, WRITE_PID = 0, 1
WRITE_OUT = 0
IN1_PTR, IN2_PTR, OUT_PTR = range(3)
LEFT, RIGHT = 0, 1

def train_addition(epochs, verbose=0):
    """
    Instantiates an Addition Core, NPI, then loads and fits model to data.

    :param epochs: Number of epochs to train for.
    """
    # Load Data
    with open(DATA_PATH_TRAIN, 'rb') as f:
        data = pickle.load(f)

    # Initialize Addition Core
    print ('Initializing Addition Core!')
    core = AdditionCore()

    # Initialize NPI Model
    print ('Initializing NPI Model!')
    npi = NPI(core, CONFIG, LOG_PATH, verbose=verbose)

    # Initialize TF Saver
    saver = tf.train.Saver()

    # Initialize TF Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Start Training
    for ep in range(1, epochs + 1):
        print(len(data))
        for i in range(len(data)):
            # Reset NPI States
            npi.reset_state()

            # Setup Environment
            in1, in2, steps = data[i]

            x, y = steps[:-1], steps[1:]
            # Run through steps, and fit!
            step_def_loss, step_arg_loss, term_acc, prog_acc, = 0.0, 0.0, 0.0, 0.0
            arg0_acc, arg1_acc, arg2_acc, num_args = 0.0, 0.0, 0.0, 0

            # dsl = DSL([], [])

            for j in range(len(x)):
                prog_name, prog_in_id, arg, term = x[j]["prog"]["command"], x[j]["prog"]["id"], x[j]["prog"]["arg"], x[j]["prog"]["terminate"]
                prog_name_out, prog_out_id, arg_out, term_out = y[j]["prog"]["command"], y[j]["prog"]["id"], y[j]["prog"]["arg"], y[j]["prog"]["terminate"]
                # Get Environment, Argument Vectors
                env_in = [x[j]["env"]]

                arg_in, arg_out = [get_args(arg, arg_in=True)], get_args(arg_out, arg_in=False)
                term_out = [1] if term_out else [0]

                # if prog_name_out=="WRITE":
                #     prog_out_id = dsl.get_code(y[j]["prog"]["arg"][1])
                #     os._exit()

                prog_in, prog_out = [[prog_in_id]], [prog_out_id]

                # Fit!
                if True:
                    t_acc, p_acc, _, loss = sess.run(
                        [npi.t_metric, npi.p_metric, npi.default_train_op, npi.default_loss],
                        feed_dict={npi.env_in: env_in, npi.arg_in: arg_in, npi.prg_in: prog_in,
                                   npi.y_prog: prog_out, npi.y_term: term_out})
                    # print({npi.prg_in: prog_in, npi.y_prog: prog_out, npi.y_term: term_out})
                    # print({npi.y_args[0]: [arg_out[0]], npi.y_args[1]: [arg_out[1]], npi.y_args[2]: [arg_out[2]]})
                    # step_arg_loss += loss
                    term_acc += t_acc
                    prog_acc += p_acc
                    step_def_loss += loss
                    # arg0_acc += a_acc[0]
                    # arg1_acc += a_acc[1]
                    # arg2_acc += a_acc[2]
                    # num_args += 1
                    # else:
                    #     loss, t_acc, p_acc, _ = sess.run(
                    #         [npi.default_loss, npi.t_metric, npi.p_metric, npi.default_train_op],
                    #         feed_dict={npi.env_in: env_in, npi.arg_in: arg_in, npi.prg_in: prog_in,
                    #                    npi.y_prog: prog_out, npi.y_term: term_out})
                    #     step_def_loss += loss
                    #     term_acc += t_acc
                    #     prog_acc += p_acc

            print ("Epoch {0:02d} Step {1:03d} Loss: {2:03f} Term: {3:03f}, Prog: {4:03f}" \
                    .format(ep, i, step_arg_loss / len(x), term_acc / len(x), prog_acc / len(x)))

        # Save Model
        saver.save(sess, CKPT_PATH)
        tf.train.write_graph(sess.graph_def, '/tmp/tf/log', 'graph.pb')