from bench_func import BenchFunction
from bench_argument import BenchScalarArgument, BenchArrayArgument
from bench_generator import BenchGenerator
import numpy as np


class PyBench:
    def __init__(self, function, input_size_list, arg_name_list):
        self.function = function
        self.input_size_list = input_size_list
        self.arg_name_list = arg_name_list
        if len(arg_name_list) != len(input_size_list):
            raise Exception("Argument names must have the same length as array sizes")
        self.bench_result = None

    def bench(self, iterations, prefix):
        """
        This function will call the function for the given
        input list and for each size we will have the defined number of iterations.
        The results will be saved or/and plotted with the given prefix.
        """
        if isinstance(self.input_size_list[0], list):
            # more than one input arrays
            turns = len(self.input_size_list[0])
        else:
            # only one input array
            turns = len(self.input_size_list)

        self.bench_result = np.zeros((turns, iterations), np.float64)
        for t in range(turns):
            for i in range(iterations):
                self.bench_result[t][i] = self.function.bench(self.input_size_list, turn=t)
        print(self.bench_result)






