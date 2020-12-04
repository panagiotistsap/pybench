from .bench_func import BenchFunction
from .bench_argument import BenchScalarArgument, BenchArrayArgument
from .bench_generator import BenchGenerator
import numpy as np
from joblib import dump


class PyBench:
    def __init__(self, function, input_size_list, arg_name_list):
        self.function = function
        self.input_size_list = input_size_list
        self.arg_name_list = arg_name_list
        self.bench_result = None

    def bench(self, iterations, filename, gpu=False, save=True):
        """
        This function will call the function for the given
        input list and for each size we will have the defined number of iterations.
        The results will be saved or/and plotted with the given filename.
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
                self.bench_result[t][i] = self.function.bench(self.input_size_list, turn=t, gpu=gpu)
        print(np.mean(self.bench_result, axis=1))
        # print(np.std(self.bench_result, axis=1))
        if save:
            dump(np.mean(self.bench_result, axis=1), "results/" + filename+"_mean.pkl")
            dump(np.std(self.bench_result, axis=1), "results/" + filename+"_std.pkl")
