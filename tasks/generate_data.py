"""
generate_data.py

Core script for generating training/test addition data. First, generates random pairs of numbers,
then steps through an execution trace, computing the exact order of subroutines that need to be
called.
"""
import pickle
import pandas as pd
import numpy as np

from dsl.trace import Trace
import datetime
import tensorflow as tf
import re

def generate_addition(prefix, num_examples, command, debug, maximum, debug_every=1000):
    """
    Generates addition data with the given string prefix (i.e. 'train', 'test') and the specified
    number of examples.

    :param prefix: String prefix for saving the file ('train', 'test')
    :param num_examples: Number of examples to generate.
    """
    times = pd.date_range('2009-10-01', end='2017-12-31', freq='5min').tolist()

    data = []
    origs = []
    formats = []
    members = set()
    for i in np.random.choice(times, size=num_examples, replace=False):
        origs.append(i)
        formats.append(i.strftime("%H:%M:%S %A, %d %B %Y"))

        for m in i.strftime("%H:%M:%S %A, %d %B %Y").replace(':', ' ').replace(',', ' ').split(' '):
            members.add(m)
        print(members)
        # if debug and i % debug_every == 0:
        #     trace = Trace(orig, formed, command, True).trace
        # else:
        #     trace = Trace(orig, formed, command).trace
    # data.append(( orig, formed, trace ))
    print(data)
    with open('tasks/env/data/{}.pik'.format(prefix), 'wb') as f:
        pickle.dump(data, f)