import time


class BenchFunction:
    # arguments of a function are
    # 1. The input size lists
    # 2. Its arguments

    def __init__(self, func, *args):
        self.func = func
        self.args = args
        for arg in self.args:
            setattr(self, arg.name, arg)

    def bench(self, input_sizes, turn=0):
        # first of all we generate the input
        input_args = []
        for arg in self.args:
            arg.generator.set_parents(arg, self)
            arg.set_size(input_sizes, turn)
            curr_arg = arg.generate()
            input_args.append(curr_arg)
        start = time.time()
        self.func(*input_args)
        return time.time() - start
