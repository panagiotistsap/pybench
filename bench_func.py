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

    def bench(self, input_sizes, turn=0, inner_iterations=10, gpu=False):
        # first of all we generate the input
        kwargs = {}
        input_args = []
        for arg in self.args:
            if arg.enable:
                arg.generator.set_parents(arg, self)
                arg.set_size(input_sizes, turn)
                curr_arg = arg.generate(gpu=gpu)
                if arg.key:
                    kwargs[arg.name] = curr_arg
                input_args.append(curr_arg)
        start = time.time()
        for i in range(inner_iterations):
            self.func(*input_args, **kwargs)
        return time.time() - start
