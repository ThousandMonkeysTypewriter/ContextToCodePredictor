"""
generate_data.py

Core script for generating training/test addition data. First, generates random pairs of numbers,
then steps through an execution trace, computing the exact order of subroutines that need to be
called.
"""
import pickle

import numpy as np

from tasks.env.trace import Trace


def generate_addition(prefix, num_examples, command, debug, maximum, debug_every=1000):
    """
    Generates addition data with the given string prefix (i.e. 'train', 'test') and the specified
    number of examples.

    :param prefix: String prefix for saving the file ('train', 'test')
    :param num_examples: Number of examples to generate.
    """
    data = []
    for i in range(num_examples):
        in1 = np.random.randint(maximum/10, maximum - 1)
        in2 = np.random.randint(maximum - in1)

        mn = min (in1, in2)
        mx = max(in1, in2)

        if debug and i % debug_every == 0:
            trace = Trace(mx, mn, command, True).trace
        else:
            trace = Trace(mx, mn, command).trace
        data.append(( mn, mx, trace ))
    with open('tasks/env/data/{}.pik'.format(prefix), 'wb') as f:
        pickle.dump(data, f)