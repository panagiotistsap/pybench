import numpy as np
from bench_argument import BenchScalarArgument


class BenchGenerator:
    def __init__(self, name, *args, **kwargs):
        """name must be uniform, normal or custom"""
        if name not in ['uniform', 'normal', 'custom']:
            raise ValueError("Generator name must be uniform, normal or custom")
        self.name = name
        if name in ['uniform', 'normal']:
            self.arg_1 = args[0]
            self.arg_2 = args[1]
        else:
            if 'generator_function' not in kwargs.keys():
                raise ValueError("You must provide generator_function when creating a custom generator")
            self.generator_func = kwargs['generator_function']
            self.parent_arg = None
            self.parent_func = None
            self.static_args = kwargs.pop('static_args', [])
            self.dynamic_args = kwargs.pop('dynamic_args', [])

    def set_parents(self, parent_arg, parent_func):
        """This function will be called by the func that will use this generator"""
        self.parent_arg = parent_arg
        self.parent_func = parent_func

    def produce_dynamic_args(self):
        res = []
        for attr in self.dynamic_args:
            if attr == 'size':
                res.append(self.parent_arg.size)
            else:
                res.append(getattr(self.parent_func, attr).generate())
        return res

    def generate(self, **kwargs):
        # the default value that a generator produces is float64
        # but can be modified through kwargs
        dtype = self.parent_arg.dtype
        if self.name == 'uniform':
            return np.random.uniform(self.arg_1, self.arg_2, self.parent_arg.size).astype(dtype)
        elif self.name == 'normal':
            return np.normal(self.parent_arg.size, self.arg_1, self.arg_2).astype(dtype)
        else:
            # arguments to be passed from the parent_func
            dynamic_args = self.produce_dynamic_args()
            if self.parent_arg.__class__ == BenchScalarArgument:
                return self.generator_func(*(self.static_args + dynamic_args), **kwargs)
            else:
                return self.generator_func(*(self.static_args + dynamic_args), **kwargs)
