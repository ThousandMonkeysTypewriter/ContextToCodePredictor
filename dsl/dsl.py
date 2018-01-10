
from tasks.env.config import CONFIG
from tasks.env.config import PROGRAM_ID as P
import numpy as np
import time
import sys

IN1_PTR, IN2_PTR, OUT_PTR = range(3)
LEFT, RIGHT = 0, 1


class DSL():           # Addition Environment
    def __init__(self, start, finish, rows=CONFIG["ENVIRONMENT_ROW"], cols=CONFIG["ENVIRONMENT_COL"]):
        # Setup Internal ScratchPad
        self.rows, self.cols = rows, cols
        self.start, self.finish = start, finish
        self.scratchpad = np.zeros((self.rows, self.cols), dtype=np.int8)

        self.trace = []
        # Initialize ScratchPad In1, In2
        self.init_scratchpad(start, finish)

        # Pointers initially all start at the right
        self.in1_ptr, self.in2_ptr, self.out_ptr = self.ptrs = \
            [(x, -1) for x in range(3)]

    def transform(self):
        """
        Builds execution trace, adding individual steps to the instance variable trace. Each
        step is represented by a triple (program_id : Integer, args : List, terminate: Boolean). If
        a subroutine doesn't take arguments, the empty list is returned.
        """
        # Seed with the starting subroutine call
        self.update_trace({"command":"TRANSFORM", "id":P["TRANSFORM"], "arg":[], "terminate":False})
        self.true_ans = self.finish

        # Execute Trace
        finish = False;
        while not finish:
            self.trans1()
            finish = self.lshift()

    def update_trace(self, trace):
        env = self.get_env()
        self.trace.append({"prog":trace, "env":env})

    def init_scratchpad(self, start, finish):
        """
        Initialize the scratchpad with the given input numbers (to be added together).
        """
        lst = [start, finish]
        for inpt in range(len(lst)):
            count = -1;
            for i in lst[inpt]:
                self.scratchpad[inpt, count] = i
                count -= 1

    def done(self):
        if self.in1_ptr[1] < -self.cols:
            # print ("$")
            return True
        else:
            lst = [x[1] for x in self.ptrs]
            if len(set(lst)) == 1:
                # print ("$$")
                return sum(sum([self[x[0], :min(x[1] + 1, -1)] for x in self.ptrs])) == 0
            else:
                # print ("$$$")
                return False

    def trans1(self):
        self.update_trace({"command":"TRANS1", "id":P["TRANS1"], "arg":[self[self.in1_ptr], self[self.in2_ptr]], "terminate":False})

        self.write_out( self[self.in2_ptr])
        # Write to Output
        self.update_trace({"command": "WRITE", "id": P["WRITE"], "arg": [0,  self[self.in2_ptr]], "terminate": False})

    def write_out(self, value):
        self[self.out_ptr] = value

    def lshift(self):
        # Move Inp1 Pointer Left
        self.update_trace({"command": "MOVE_PTR", "id": P["MOVE_PTR"], "arg": [IN1_PTR, LEFT], "terminate": False})
        self.update_trace({"command": "MOVE_PTR", "id": P["MOVE_PTR"], "arg": [IN2_PTR, LEFT], "terminate": False})

        # Move Inp1 Pointer Left (check if done)
        self.update_trace({"command": "MOVE_PTR", "id": P["MOVE_PTR"], "arg": [OUT_PTR, LEFT], "terminate": False})

        self.in1_ptr, self.in2_ptr, self.out_ptr = self.ptrs = \
            [(x, y - 1) for (x, y) in self.ptrs]

        done = self.done()
        if (done):
            self.trace.append({"prog": {"command": "MOVE_PTR", "id": P["MOVE_PTR"], "arg": [0, 0], "terminate": True}, "env": {}})
        return done

    def pretty_print(self):
        new_strs = ["".join(map(str, self[i])) for i in range(3)]
        line_length = len('Input 1:' + " " * 5 + new_strs[0])
        print ('Input 1:' + " " * 5 + new_strs[0])
        print ('Input 2:' + " " * 5 + new_strs[1])
        print ('-' * line_length)
        print ('Output :' + " " * 5 + new_strs[2])
        print ('True out:' + " " * 5 + str(self.true_ans))
        print ('')
        time.sleep(.1)
        sys.stdout.flush()

    def get_env(self):
        env = np.zeros((CONFIG["ENVIRONMENT_ROW"], CONFIG["ENVIRONMENT_DEPTH"]), dtype=np.int32)
        if self.in1_ptr[1] < -CONFIG["ENVIRONMENT_COL"]:
            env[0][0] = 1
        else:
            env[0][self[self.in1_ptr]] = 1
        if self.in2_ptr[1] < -CONFIG["ENVIRONMENT_COL"]:
            env[1][0] = 1
        else:
            env[1][self[self.in2_ptr]] = 1
        if self.out_ptr[1] < -CONFIG["ENVIRONMENT_COL"]:
            env[2][0] = 1
        else:
            env[2][self[self.out_ptr]] = 1
        return env.flatten()

    def __getitem__(self, item):
        return self.scratchpad[item]

    def __setitem__(self, key, value):
        self.scratchpad[key] = value

