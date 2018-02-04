"""
config.py

Configuration Variables for the Addition NPI Task => Stores Scratch-Pad Dimensions, Vector/Program
Embedding Information, etc.
"""
import numpy as np
import tensorflow as tf

DATA_PATH_TRAIN = "tasks/env/data/train.pik"
DATA_PATH_TEST = "tasks/env/data/test.pik"
LOG_PATH = "log/"
CKPT_PATH = "log/model.ckpt"
DSL_DATA_PATH = "dsl/data/data_buffer.json"

CONFIG = {
    "ENVIRONMENT_ROW": 5,         # Input 1, Input 2, Carry, Output
    "ENVIRONMENT_COL": 5,         # 10-Digit Maximum for Addition Task
    "ENVIRONMENT_DEPTH": 74,      # Size of each element vector => One-Hot, Options: 0-9

    "ARGUMENT_NUM": 1,            # Maximum Number of Program Arguments
    "ARGUMENT_DEPTH": 75,         # Size of Argument Vector => One-Hot, Options 0-9, Default (10)
    "DEFAULT_ARG_VALUE": 74,      # Default Argument Value

    "PROGRAM_NUM": 7,             # Maximum Number of Subroutines
    "PROGRAM_KEY_SIZE": 7,        # Size of the Program Keys
    "PROGRAM_EMBEDDING_SIZE": 10  # Size of the Program Embeddings
}

PROGRAM_SET = [
    ("MOVE_PTR", 4, 2),       # Moves Pointer (4 options) either left or right (2 options)
    ("WRITE", 2, 10),         # Given Carry/Out Pointer (2 options) writes digit (10 options)
    ("TRANSFORM",),           # Top-Level Add Program (calls children routines)
    ("TRANS1",),              # Single-Digit (Column) Add Operation
    ("LSHIFT",)               # Shifts all Pointers Left (after Single-Digit Add)
]

PROGRAM_ID = {x[0]: i for i, x in enumerate(PROGRAM_SET)}

def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")

class Arguments():             # Program Arguments
    def __init__(self, args, num_args=CONFIG["ARGUMENT_NUM"], arg_depth=CONFIG["ARGUMENT_DEPTH"]):
        self.args = args
        self.arg_vec = np.zeros((num_args, arg_depth), dtype=np.float32)

def get_args(args, arg_in=True):
    if arg_in:
        arg_vec = np.zeros((CONFIG["ARGUMENT_NUM"], CONFIG["ARGUMENT_DEPTH"]), dtype=np.int32)
    else:
        arg_vec = [np.zeros((CONFIG["ARGUMENT_DEPTH"]), dtype=np.int32) for _ in
                   range(CONFIG["ARGUMENT_NUM"])]
    # if len(args) > 0:
    #     for i in range(CONFIG["ARGUMENT_NUM"]):
    #         if i >= len(args):
    #             arg_vec[i][CONFIG["DEFAULT_ARG_VALUE"]] = 1
    #         else:
    #             arg_vec[i][args[i]] = 1
    # else:
    for i in range(CONFIG["ARGUMENT_NUM"]):
        arg_vec[i][CONFIG["DEFAULT_ARG_VALUE"]] = 1
    return arg_vec.flatten() if arg_in else arg_vec

def get_env(data):
    env = np.zeros((CONFIG["ENVIRONMENT_ROW"], CONFIG["ENVIRONMENT_DEPTH"]), dtype=np.int32)

    # print(data)
    env[0][data["answer"]] = 1
    env[1][data["output"]] = 1
    env[2][data["period"]] = 1
    env[3][data["client_id"]] = 1

    return env.flatten()


