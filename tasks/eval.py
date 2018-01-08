"""
eval.py

Loads in an Addition NPI, and starts a REPL for interactive addition.
"""
from model.npi import NPI
from tasks.env.addition import AdditionCore
from tasks.env.config import CONFIG, get_args, PROGRAM_SET, LOG_PATH, DATA_PATH, CKPT_PATH
import dsl.dsl
import numpy as np
import pickle
import tensorflow as tf
from dsl.dsl import ScratchPad

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
        with open(DATA_PATH, 'rb') as f:
            data = pickle.load(f)

        # Initialize Addition Core
        core = AdditionCore()

        # Initialize NPI Model
        npi = NPI(core, CONFIG, LOG_PATH)

        # Restore from Checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, CKPT_PATH)

        # Run REPL
        eq = 0
        not_eq=0
        for x in range(0, 1):
            res = ""
            # try:
            res = repl(sess, npi, data, "TRANSFORM")
            # except:
            #     print ("")
            if res:
               eq+=1
            else:
               not_eq+=1
        print(eq, not_eq)
        # repeat()

def repl(session, npi, data, command):
        x, y, steps = data[3]

        f = open('log/numbers.txt', 'r+')
        f.truncate()

        f = open('log/prog_orig.txt', 'r+')
        f.truncate()

        f = open('log/prog_produced.txt', 'r+')
        f.truncate()

        with open("log/numbers.txt", "a") as myfile:
            myfile.write(str(x)+","+str(y) + "\n")

        with open("log/prog_orig.txt", "a") as myfile:
            myfile.write(str(steps))

        # Reset NPI States
        print ("")
        npi.reset_state()

        # Setup Environment
        true_ans = y;

        scratch = ScratchPad(x, y, true_ans)

        prog_name, prog_id, term =  command, 2, False

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

            print ('Step: %s, Arguments: %s, Terminate: %s' % (prog_name, a_str, str(term)))
            # print 'IN 1: %s, IN 2: %s, CARRY: %s, OUT: %s' % (scratch.in1_ptr[1],
            #                                                   scratch.in2_ptr[1],
            #                                                   scratch.carry_ptr[1],
            #                                                   scratch.out_ptr[1])

            # Update Environment if MOVE or WRITE
            if prog_id == MOVE_PID or prog_id == WRITE_PID:
                scratch.execute(prog_id, arg)

            # Print Environment
            # scratch.pretty_print()

            # Get Environment, Argument Vectors
            # Current step
            env_in, arg_in, prog_in = [scratch.get_env()], [get_args(arg, arg_in=True)], [[prog_id]]
            # print (env_in, arg, prog_in)
            t, n_p, n_args = session.run([npi.terminate, npi.program_distribution, npi.arguments],
                                         feed_dict={npi.env_in: env_in, npi.arg_in: arg_in,
                                                    npi.prg_in: prog_in})

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

                trace_ans = []
                for i in scratch[2]:
                    trace_ans.insert(0, i)
                # print ("Input:  %s, %s, Output:  %s, %s" % (str(x), str(y), str(output), scratch.true_ans))
                with open("log/prog_produced.txt", "a") as myfile:
                    myfile.write(str(prog_id) + "," + str(np.argmax(n_args[0])) + "," + str(np.argmax(n_args[1])) + ", terminate: true\n")
                return True

            else:
                prog_id = np.argmax(n_p)
                prog_name = PROGRAM_SET[prog_id][0]
                term = False
                print([np.argmax(n_p), PROGRAM_SET[prog_id][0]], [np.argmax(n_args[0]), np.argmax(n_args[1])])
                with open("log/prog_produced.txt", "a") as myfile:
                    myfile.write(str(prog_id) + "," + str(np.argmax(n_args[0])) + "," + str(np.argmax(n_args[1])) + ","+str(np.argmax(t))+"\n")

            # cont = raw_input('Continue? ')

def repeat():
        with open("log/numbers.txt", 'r') as f:
            fl = f.readline()
            x, y = fl.rstrip('\n').split(",")
            # print(x,y)
            scratch = ScratchPad(x, y, int(x)-int(y))

        lines = [line.rstrip('\n') for line in open("log/prog.txt")]

        for c in lines:
            prog_id, arg0, arg1 = map(int, c.rstrip('\n').split(","))

            if prog_id == MOVE_PID or prog_id == WRITE_PID:
                scratch.execute(prog_id, [arg0, arg1])
            # Print Environment
            scratch.pretty_print()
