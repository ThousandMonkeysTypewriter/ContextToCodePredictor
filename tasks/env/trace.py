"""
trace.py

Core class definition for a trace object => given a pair of integers to add, builds the execution
trace, calling the specified subprograms.
"""
from tasks.env.config import ScratchPad, PROGRAM_ID as P
ADD, ADD1, WRITE, LSHIFT, CARRY, MOVE_PTR, REDUCE, REDUCE1 = "ADD", "ADD1", "WRITE", "LSHIFT", "CARRY", "MOVE_PTR", "REDUCE", "REDUCE1"
WRITE_OUT, WRITE_CARRY = 0, 1
IN1_PTR, IN2_PTR, CARRY_PTR, OUT_PTR = range(4)
LEFT, RIGHT = 0, 1


class Trace():
    def __init__(self, in1, in2, command, debug=False):
        """
        Instantiates a trace object, and builds the exact execution pipeline for adding the given
        parameters.
        """
        self.in1, self.in2, self.debug = in1, in2, debug

        self.trace = []
        # Check answer
        if command == "ADD":
            # Build Execution Trace
            scratch = ScratchPad(in1, in2, self.in1 + self.in2)
            self.scratch = scratch
            self.build_add()

        elif command == "REDUCE":
            # Build Execution Trace
            scratch = ScratchPad(in1, in2, self.in1 - self.in2)
            self.scratch = scratch
            self.build_reduce()

        trace_ans = int("".join(map(str, map(int, self.scratch[3]))))

        assert(scratch.true_ans == trace_ans), "%s not equals %s in %s %s %s" % (true_ans, trace_ans, in1, command, in2)

    def build_add(self):
        """
        Builds execution trace, adding individual steps to the instance variable trace. Each
        step is represented by a triple (program_id : Integer, args : List, terminate: Boolean). If
        a subroutine doesn't take arguments, the empty list is returned.
        """
        # Seed with the starting subroutine call
        self.trace.append(((ADD, P[ADD]), [], False))

        # Execute Trace
        while not self.scratch.done():
            self.add1()
            self.lshift()

    def build_reduce(self):
        """
        Builds execution trace, adding individual steps to the instance variable trace. Each
        step is represented by a triple (program_id : Integer, args : List, terminate: Boolean). If
        a subroutine doesn't take arguments, the empty list is returned.
        """
        # Seed with the starting subroutine call
        self.trace.append(((REDUCE, P[REDUCE]), [], False))

        # Execute Trace
        while not self.scratch.done():
            self.reduce1()
            self.lshift()

    def add1(self):
        # Call Add1 Subroutine
        self.trace.append(( (REDUCE1, P[REDUCE1]), [], False ))
        out, carry = self.scratch.add1()

        # Write to Output
        self.trace.append(( (WRITE, P[WRITE]), [WRITE_OUT, out], False ))
        self.scratch.write_out(out, self.debug)

        # Carry Condition
        if carry > 0:
            self.carry(carry)

    def reduce1(self):
        # Call Add1 Subroutine
        self.trace.append(( (REDUCE1, P[REDUCE1]), [], False ))
        out, carry = self.scratch.reduce1()

        # Write to Output
        self.trace.append(( (WRITE, P[WRITE]), [WRITE_OUT, out], False ))
        self.scratch.write_out(out, self.debug)

        # Carry Condition
        if carry > 0:
            self.carry(carry)

    def carry(self, carry_val):
        # Call Carry Subroutine
        self.trace.append(( (CARRY, P[CARRY]), [], False ))

        # Shift Carry Pointer Left
        self.trace.append(( (MOVE_PTR, P[MOVE_PTR]), [CARRY_PTR, LEFT], False ))

        # Write Carry Value
        self.trace.append(( (WRITE, P[WRITE]), [WRITE_CARRY, carry_val], False ))

        # Shift Carry Pointer Right
        self.trace.append(( (MOVE_PTR, P[MOVE_PTR]), [CARRY_PTR, RIGHT], False ))

        # Perform Carry Logic on Scratchpad
        self.scratch.write_carry(carry_val, self.debug)

    def lshift(self):
        # Perform LShift Logic on Scratchpad
        self.scratch.lshift()

        # Move Inp1 Pointer Left
        self.trace.append(( (MOVE_PTR, P[MOVE_PTR]), [IN1_PTR, LEFT], False ))

        # Move Inp1 Pointer Left
        self.trace.append(( (MOVE_PTR, P[MOVE_PTR]), [IN2_PTR, LEFT], False ))

        # Move Inp1 Pointer Left
        self.trace.append(( (MOVE_PTR, P[MOVE_PTR]), [CARRY_PTR, LEFT], False ))

        # Move Inp1 Pointer Left (check if done)
        if self.scratch.done():
            self.trace.append(( (MOVE_PTR, P[MOVE_PTR]), [OUT_PTR, LEFT], True ))
        else:
            self.trace.append(( (MOVE_PTR, P[MOVE_PTR]), [OUT_PTR, LEFT], False ))