"""
eval.py

Loads in an Addition NPI, and starts a REPL for interactive addition.
"""
from model.npi import NPI
from tasks.env.addition import AdditionCore
from tasks.env.config import CONFIG, get_args, PROGRAM_SET, LOG_PATH, DATA_PATH_TEST, CKPT_PATH
from tasks.env.config import get_env
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.platform import gfile

MOVE_PID, WRITE_PID = 0, 1
R_L = {0: "LEFT", 1: "RIGHT"}


def evaluate_addition():
    """
    Load NPI Model from Checkpoint, and initialize REPL, for interactive carry-addition.
    """
    with tf.Session() as sess:
        # Load Data
        with open(DATA_PATH_TEST, 'rb') as f:
            data = pickle.load(f)

        # Initialize Addition Core
        core = AdditionCore()

        # Initialize NPI Model
        npi = NPI(core, CONFIG, LOG_PATH)

        # Restore from Checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, CKPT_PATH)

        # with gfile.FastGFile("/tmp/tf/log/graph.pb", 'rb') as f:
        #     graph_def = tf.GraphDef()
        #     graph_def.ParseFromString(f.read())
        #     sess.graph.as_default()
        #     tf.import_graph_def(graph_def)
        # print("map variables")

        # Run REPL
        for x in range(0, 10):
            res = ""
            # try:
            repl(sess, npi, data, x)
            # except:
            print ("--------------------------")
            # if res:
            #    eq+=1
            # else:
            #    not_eq+=1
        # repeat()

def repl(session, npi, data, pos):
        steps = data[pos]
        f = open('log/numbers.txt', 'r+')
        f.truncate()

        f = open('log/prog_orig.txt', 'r+')
        f.truncate()

        f = open('log/prog_produced.txt', 'r+')
        f.truncate()

        with open("log/prog_orig.txt", "a") as myfile:
            for s in steps:
                myfile.write(str(s)+"\n")

        # Reset NPI States
        npi.reset_state()

        count = 0

        x, y = steps[:-1], steps[1:]

        for j in range(len(x)):
            if count == 0:
                print ('y = Prog_id: %s, Terminate: %d' % (x[j]["program"]["id"], x[j]["environment"]["terminate"]))
                print ('y` = Prog_id: %s, Terminate: %d' % (x[j]["program"]["id"], x[j]["environment"]["terminate"]))

            prog_in, arg_in = [[x[j]["program"]["id"]]],  [get_args(x[j]["args"]["id"], arg_in=True)]
            prog_out, terminate_out = y[j]["program"]["id"],  y[j]["environment"]["terminate"]
            env_in = [get_env(x[j]["environment"])]

            # print (env_in, arg, prog_in)
            t, n_p = session.run([npi.terminate, npi.program_distribution],
                                         feed_dict={npi.env_in: env_in, npi.arg_in:arg_in, npi.prg_in: prog_in})

            prog_id = np.argmax(n_p)

            print ('y= Prog_id: %s, Terminate: %d' % (prog_id, np.argmax(t)))
            print ('y` = Prog_id: %s, Terminate: %d' % (prog_out, terminate_out))

            count += 1

            # Next step
            if np.argmax(t) == 1:
                # print 'Step: %s, Arguments: %s, Terminate: %s' % (prog_name, a_str, str(True))
                # print 'IN 1: %s, IN 2: %s, CARRY: %s, OUT: %s' % (scratch.in1_ptr[1],
                #                                                   scratch.in2_ptr[1],
                #                                                   scratch.carry_ptr[1],
                #                                                   scratch.out_ptr[1])
                # Update Environment if MOVE or WRITE
                # if prog_id == MOVE_PID or prog_id == WRITE_PID:
                #     scratch.execute(prog_id, arg)

                # print ("Input:  %s, %s, Output:  %s, %s" % (str(x), str(y), str(output), scratch.true_ans))
                with open("log/prog_produced.txt", "a") as myfile:
                    myfile.write(str(prog_id) + ", terminate: true\n")
                return True

            else:
                # prog_name = PROGRAM_SET[prog_id][0]

                # print([np.argmax(n_p), PROGRAM_SET[prog_id][0]], [np.argmax(n_args[0]), np.argmax(n_args[1])])
                with open("log/prog_produced.txt", "a") as myfile:
                    myfile.write(str(prog_id) + ","+str(np.argmax(t))+"\n")

            # cont = raw_input('Continue? ')

def repeat():
        lines = [line.rstrip('\n') for line in open("log/prog.txt")]

        for c in lines:
            prog_id, arg0, arg1 = map(int, c.rstrip('\n').split(","))

            if prog_id == MOVE_PID or prog_id == WRITE_PID:
                scratch.execute(prog_id, [arg0, arg1])
            # Print Environment
            scratch.pretty_print()
