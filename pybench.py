from bench_func import BenchFunction
from bench_argument import BenchScalarArgument, BenchArrayArgument
from bench_generator import BenchGenerator


class PyBench:
    def __init__(self, function, input_size_list):
        self.function = function
        self.input_size_list = input_size_list

    # def bench(self, iterations):



