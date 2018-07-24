"""
generate_data.py

Core script for generating training/test addition data. First, generates random pairs of numbers,
then steps through an execution trace, computing the exact order of subroutines that need to be
called.
"""
import pickle
import pandas as pd
import numpy as np

from dsl.dsl import DSL
import datetime
import tensorflow as tf
import re
import json
from tasks.env.config import DSL_DATA_PATH
from pprint import pprint
import collections

def explode (str):
    return str.replace(':', ' ').replace(', ', ' ').replace('-', ' ').split(' ')

def exec_ (orig, formatted):
    dsl = DSL(orig, formatted)
    dsl.transform()

    trace_ans = []
    for i in dsl[2]:
        trace_ans.insert(0, i)

    assert (str(dsl.true_ans) == str(trace_ans)), "%s not equals %s in %s %s" % (
        dsl.true_ans, trace_ans, orig, formatted)
    return dsl.trace

def generate_addition( ):
    """
    Generates addition data with the given string prefix (i.e. 'train', 'test') and the specified
    number of examples.

    :param prefix: String prefix for saving the file ('train', 'test')
    :param num_examples: Number of examples to generate.
    """

    with open(DSL_DATA_PATH, 'r') as handle:
        parsed = json.load(handle)

    # times = pd.date_range('2000-10-01', end='2017-12-31', freq='5min').tolist()
    #
    train_data = []
    test_data = []
    count = 0
    # dates = []
    # members_set = set()
    # for i in np.random.choice(times, size=num_examples, replace=False):
    #     # key = i.strftime("%Y-%m-%d %H:%M:%S")
    #     # value = i.strftime("%H:%M:%S %A, %d %B %Y")
    #
    #     key = i.strftime("y%Y m%m d%d")
    #     value = i.strftime("d%d m%B y%Y")
    #
    #     # key = i.strftime("m%m 0 0")
    #     # value = i.strftime("m%B 0 0")
    #
    #     dates.append({"k":key, "v":value})
    #
    #     for m in explode(value):
    #         members_set.add(m)
    #     for m in explode(key):
    #         members_set.add(m)
    # members_list = list(members_set)
    # count = 0
    for row_r in parsed:
        count += 1
        row = collections.OrderedDict(sorted(row_r.items()))
        trace = []
        for key, values in row.items():
            step = {}
        # count += 1
        # key_list = []
        # value_list = []
        # for k in explode(d["k"]):
        #     key_list.append(members_list.index(k))
        #
        # for v in explode(d["v"]):
        #     value_list.append(members_list.index(v))
        #
        # trace = exec_( key_list, value_list )
        #
        # if debug and count % debug_every == 0:
        #     print(trace)
        #     if (k == "terminate"):
        #     print(key)
            for k, v in values.items():
                if k == 'supervised_env':
                    environment = {}
                    for e_k, e_v in v.items():
                        if e_k == 'terminate':
                            environment['terminate'] = e_v.get('value')
                        elif e_k == 'answer':
                            environment['answer'] = e_v.get('value')
                        elif e_k == 'output':
                            environment['output'] = e_v.get('value')
                        elif e_k == 'date2':
                            environment['date2'] = e_v.get('value')
                        elif e_k == 'date2_diff':
                            environment['date2_diff'] = e_v.get('value')
                        elif e_k == 'date1':
                            environment['date1'] = e_v.get('value')
                        elif e_k == 'date1_diff':
                            environment['date1_diff'] = e_v.get('value')
                        elif e_k == 'client_id':
                            environment['client_id'] = e_v.get('value')
                    step['environment'] = environment
                elif k == 'argument':
                    args = {}
                    for e_k, e_v in v.items():
                        if e_k == 'id':
                            args['id'] = e_v.get('value')
                    step['args'] = args
                elif k == 'program':
                    program = {}
                    for e_k, e_v in v.items():
                        if e_k == 'program':
                            program['program'] = e_v.get('value')
                        if e_k == 'id':
                            program['id'] = e_v.get('value')

                    step['program'] = program
            trace.append(step)
        print(trace)
                    # ({"command": "MOVE_PTR", "id": P["MOVE_PTR"], "arg": [OUT_PTR, LEFT], "terminate": False})
        if (count % 5==0):
            test_data.append(trace)
        else:
            train_data.append(trace)
    with open('tasks/env/data/test.pik', 'wb') as f:
        pickle.dump(test_data, f)
    with open('tasks/env/data/train.pik', 'wb') as f:
        pickle.dump(train_data, f)
    # with open('tasks/env/data/train.pik1', 'a') as f:
    #     for c in train_data:
    #         f.write(str(c))