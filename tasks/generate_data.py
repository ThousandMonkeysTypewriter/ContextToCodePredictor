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

def explode (str):
    return str.replace(':', ' ').replace(', ', ' ').replace('-', ' ').split(' ')

def generate_addition(prefix, num_examples, command, debug, maximum, debug_every=1000):
    """
    Generates addition data with the given string prefix (i.e. 'train', 'test') and the specified
    number of examples.

    :param prefix: String prefix for saving the file ('train', 'test')
    :param num_examples: Number of examples to generate.
    """
    times = pd.date_range('2009-10-01', end='2017-12-31', freq='5min').tolist()

    data = []
    dates = {}
    members_set = set()
    for i in np.random.choice(times, size=num_examples, replace=False):
        # key = i.strftime("%Y-%m-%d %H:%M:%S")
        # value = i.strftime("%H:%M:%S %A, %d %B %Y")

        key = i.strftime("%Y %m %d")
        value = i.strftime("%d %B %Y")

        dates[key] = value

        for m in explode(value):
            members_set.add(m)
        for m in explode(key):
            members_set.add(m)
    members_list = list(members_set)
    count = 0
    for key, value in dates.items():
        count += 1
        key_list = []
        value_list = []
        for k in explode(key):
            key_list.append(members_list.index(k))

        for v in explode(value):
            value_list.append(members_list.index(v))

        print(value_list, key_list, value, key)
        if debug and count % debug_every == 0:
            trace = Trace(key_list, value_list, command, True).trace
        else:
            trace = Trace(key_list, value_list, command).trace
    data.append(( orig, formed, trace ))

    with open('tasks/env/data/{}.pik'.format(prefix), 'wb') as f:
        pickle.dump(data, f)