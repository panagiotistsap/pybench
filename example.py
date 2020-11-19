from pybench import *
import numpy as np


# def func(a, b):
#     print(a+b)
#     return a + b
#
#
# def custom_generator(a):
#     return a
#
#
# def custom_array_generator(size):
#     return [x for x in range(size)]
#
#
# bench_func = BenchFunction(
#     func,
#     BenchScalarArgument('a', np.float64,
#                         BenchGenerator('custom',
#                                        generator_function=custom_generator)),
#     BenchScalarArgument('b', np.float64,
#                         BenchGenerator('custom',
#                                        generator_function=custom_generator))
#     )
#
# bench_func = BenchFunction(
#     func,
#     BenchArrayArgument('a', np.float64,
#                        BenchGenerator('custom',
#                                       generator_function=custom_array_generator)),
#     BenchArrayArgument('b', np.float64,
#                        BenchGenerator('custom',
#                                       generator_function=custom_array_generator))
#     )
# bench_func.bench([1, 2, 3, 4, 5])
#

# complex function
def produce_bin_centers(slices, beam_dt):
    min_beam_dt = min(beam_dt)
    max_beam_dt = max(beam_dt)
    bin_width = (max_beam_dt-min_beam_dt)//slices
    bin_centers = np.linspace(min_beam_dt+bin_width//2, max_beam_dt-bin_width//2, slices)
    return bin_centers


def lickick(dt, bin_centers):
    print(len(dt))
    print(len(bin_centers))


bench_likick = BenchFunction(
    lickick,
    BenchArrayArgument('dt', np.float64,
                       BenchGenerator('uniform',
                                      4, 4)),
    BenchArrayArgument('bin_centers', np.float64,
                       BenchGenerator('custom',
                                      generator_function=produce_bin_centers,
                                      dynamic_args=['size', 'dt']),
                       index=1)
)

bench_likick.bench([[1000], [10]])