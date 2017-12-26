"""
eval.py

Loads in an Addition NPI, and starts a REPL for interactive addition.
"""
from model.npi import NPI
from tasks.env.addition import AdditionCore
from tasks.env.config import CONFIG, get_args, PROGRAM_SET, ScratchPad, LOG_PATH, DATA_PATH, CKPT_PATH
import numpy as np
import pickle
import tensorflow as tf

MOVE_PID, WRITE_PID = 0, 1
W_PTRS = {0: "OUT", 1: "CARRY"}
PTRS = {0: "IN1_PTR", 1: "IN2_PTR", 2: "CARRY_PTR", 3: "OUT_PTR"}
R_L = {0: "LEFT", 1: "RIGHT"}


def evaluate_addition():
    """
    Load NPI Model from Checkpoint, and initialize REPL, for interactive carry-addition.
    """
    with tf.Session() as sess:
        # Load Data
        with open(DATA_PATH, 'r') as f:
            data = pickle.load(f)

        # Initialize Addition Core
        core = AdditionCore()

        # Initialize NPI Model
        npi = NPI(core, CONFIG, LOG_PATH)

        # Restore from Checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, CKPT_PATH)

        # Run REPL
        repl(sess, npi, data)
        repeat()

def repl(session, npi, data):
        inpt = raw_input('Enter Numbers, or Hit Enter for Random Pair: ')

        if inpt == "":
            x, y, _ = data[np.random.randint(len(data))]

        else:
            x, y = map(int, inpt.split())

        f = open('log/numbers.txt', 'r+')
        f.truncate()

        f = open('log/prog.txt', 'r+')
        f.truncate()

        with open("log/numbers.txt", "a") as myfile:
            myfile.write(str(x)+","+str(y) + "\n")

        # Reset NPI States
        print ""
        npi.reset_state()

        # Setup Environment
        scratch = ScratchPad(x, y)
        prog_name, prog_id, term = 'REDUCE', 2, False

        cont = 'c'

        while cont == 'c' or cont == 'C':
            #Previous step
            if prog_id == MOVE_PID or prog_id == WRITE_PID:
                arg = [np.argmax(n_args[0]), np.argmax(n_args[1])]
            else:
                arg = []

            # Print Step Output
            if prog_id == MOVE_PID:
                a0, a1 = PTRS.get(arg[0], "OOPS!"), R_L[arg[1]]
                a_str = "[%s, %s]" % (str(a0), str(a1))
            elif prog_id == WRITE_PID:
                a0, a1 = W_PTRS[arg[0]], arg[1]
                a_str = "[%s, %s]" % (str(a0), str(a1))
            else:
                a_str = "[]"

            print 'Step: %s, Arguments: %s, Terminate: %s' % (prog_name, a_str, str(term))
            print 'IN 1: %s, IN 2: %s, CARRY: %s, OUT: %s' % (scratch.in1_ptr[1],
                                                              scratch.in2_ptr[1],
                                                              scratch.carry_ptr[1],
                                                              scratch.out_ptr[1])

            # Update Environment if MOVE or WRITE
            if prog_id == MOVE_PID or prog_id == WRITE_PID:
                scratch.execute(prog_id, arg)

            if arg:
                with open("log/prog.txt", "a") as myfile:
                    myfile.write(str(prog_id) + "," + str(np.argmax(n_args[0])) + "," + str(np.argmax(n_args[1])) + "\n")

            # Print Environment
            scratch.pretty_print()

            # Get Environment, Argument Vectors
            # Current step
            env_in, arg_in, prog_in = [scratch.get_env()], [get_args(arg, arg_in=True)], [[prog_id]]
            t, n_p, n_args = session.run([npi.terminate, npi.program_distribution, npi.arguments],
                                         feed_dict={npi.env_in: env_in, npi.arg_in: arg_in,
                                                    npi.prg_in: prog_in})

            # Next step
            if np.argmax(t) == 1:
                print 'Step: %s, Arguments: %s, Terminate: %s' % (prog_name, a_str, str(True))
                print 'IN 1: %s, IN 2: %s, CARRY: %s, OUT: %s' % (scratch.in1_ptr[1],
                                                                  scratch.in2_ptr[1],
                                                                  scratch.carry_ptr[1],
                                                                  scratch.out_ptr[1])
                # Update Environment if MOVE or WRITE
                # if prog_id == MOVE_PID or prog_id == WRITE_PID:
                #     scratch.execute(prog_id, arg)

                output = int("".join(map(str, map(int, scratch[3]))))
                print "Model Output: %s + %s = %s" % (str(x), str(y), str(output))
                print "Correct Out : %s + %s = %s" % (str(x), str(y), str(x + y))
                print "Correct!" if output == (x + y) else "Incorrect!"
                cont = "n"

            else:
                prog_id = np.argmax(n_p)
                prog_name = PROGRAM_SET[prog_id][0]
                term = False

            # cont = raw_input('Continue? ')

def repeat():
        with open("log/numbers.txt", 'r') as f:
            fl = f.readline()
            x, y = fl.rstrip('\n').split(",")
            # print(x,y)
            scratch = ScratchPad(x, y)

        lines = [line.rstrip('\n') for line in open("log/prog.txt")]

        for c in lines:
            prog_id, arg0, arg1 = map(int, c.rstrip('\n').split(","))

            if prog_id == MOVE_PID or prog_id == WRITE_PID:
                scratch.execute(prog_id, [arg0, arg1])
            # Print Environment
            scratch.pretty_print()