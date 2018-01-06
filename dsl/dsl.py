
from tasks.env.config import CONFIG
import numpy as np
import time
import sys


class ScratchPad():           # Addition Environment
    def __init__(self, start, finish, true_ans_, rows=CONFIG["ENVIRONMENT_ROW"], cols=CONFIG["ENVIRONMENT_COL"]):
        # Setup Internal ScratchPad
        self.rows, self.cols = rows, cols
        self.true_ans = true_ans_
        self.scratchpad = np.zeros((self.rows, self.cols), dtype=np.int8)

        # Initialize ScratchPad In1, In2
        self.init_scratchpad(start, finish)

        # Pointers initially all start at the right
        self.in1_ptr, self.in2_ptr, self.out_ptr = self.ptrs = \
            [(x, -1) for x in range(3)]

        # self.out_ptr = [(x, -1) for x in range(4)]

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
            return True
        else:
            lst = [x[1] for x in self.ptrs]
            if len(set(lst)) == 1:
                return sum(sum([self[x[0], :min(x[1] + 1, -1)] for x in self.ptrs])) == 0
            else:
                return False

    def trans1(self):
        temp = self[self.in2_ptr]
        return temp

    def reduce1(self):
        temp = self[self.in1_ptr] - self[self.in2_ptr] - self[self.carry_ptr]
        if temp < 0:
            carry = 1
        else:
            carry = 0;
        return temp % 10, carry

    def write_carry(self, carry_val, debug=False):
        carry_row, carry_col = self.carry_ptr
        self[(carry_row, carry_col - 1)] = carry_val
        if debug:
            self.pretty_print()

    def write_out(self, value, debug=False):
        self[self.out_ptr] = value
        if debug:
            self.pretty_print()

    def lshift(self):
        self.in1_ptr, self.in2_ptr, self.out_ptr = self.ptrs = \
            [(x, y - 1) for (x, y) in self.ptrs]

    def pretty_print(self):
        new_strs = ["".join(map(str, self[i])) for i in range(4)]
        line_length = len('Input 1:' + " " * 5 + new_strs[0])
        print ('Input 1:' + " " * 5 + new_strs[0])
        print ('Input 2:' + " " * 5 + new_strs[1])
        print ('Carry  :' + " " * 5 + new_strs[2])
        print ('-' * line_length)
        print ('Output :' + " " * 5 + new_strs[3])
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
        if self.carry_ptr[1] < -CONFIG["ENVIRONMENT_COL"]:
            env[2][0] = 1
        else:
            env[2][self[self.carry_ptr]] = 1
        if self.out_ptr[1] < -CONFIG["ENVIRONMENT_COL"]:
            env[3][0] = 1
        else:
            env[3][self[self.out_ptr]] = 1
        return env.flatten()

    def execute(self, prog_id, args):
        if prog_id == 0:               # MOVE!
            ptr, lr = args
            lr = (lr * 2) - 1

            if ptr == 0:
                self.in1_ptr = (self.in1_ptr[0], self.in1_ptr[1] + lr)
            elif ptr == 1:
                self.in2_ptr = (self.in2_ptr[0], self.in2_ptr[1] + lr)
            elif ptr == 2:
                self.carry_ptr = (self.carry_ptr[0], self.carry_ptr[1] + lr)
            elif ptr == 3:
                self.out_ptr = (self.out_ptr[0], self.out_ptr[1] + lr)
            else:
                raise NotImplementedError
            self.ptrs = [self.in1_ptr, self.in2_ptr, self.carry_ptr, self.out_ptr]
        elif prog_id == 1:             # WRITE!
            ptr, val = args
            if ptr == 0:
                self[self.out_ptr] = val
            elif ptr == 1:
                self[self.carry_ptr] = val
            else:
                raise NotImplementedError

    def __getitem__(self, item):
        return self.scratchpad[item]

    def __setitem__(self, key, value):
        self.scratchpad[key] = value

