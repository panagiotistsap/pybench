import blond.utils.bmath as bm
import pybench.main as pb
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Benchmarking Functions.')
parser.add_argument('--gpu', '-g', type=str, choices=['True', 'False'], default="False")
parser.add_argument('--precision', '-p', choices=['single', 'double'], default='double')
args = parser.parse_args()

device = "cpu"
bm.use_precision(args.precision)
profile_dtype = np.float64
gpu = False
if args.gpu == "True":
    bm.use_gpu()
    device = "gpu"
    profile_dtype = np.int32
    gpu = True
solver = "simple"
alpha_order = 0


def get_solver():
    global solver
    return solver.encode('utf-8')


def get_alpha_order():
    return alpha_order


bf_histo = pb.BenchFunction(
    bm.slice,
    pb.BenchArrayArgument('dt', bm.precision.real_t,
                          pb.BenchGenerator('normal', 10 ** (-6), 10 ** (-12)), index=0),
    pb.BenchArrayArgument('profile', profile_dtype,
                          pb.BenchGenerator('custom', generator_function=np.zeros, dynamic_args=['size']), index=1),
    pb.BenchScalarArgument('cut_left', bm.precision.real_t,
                           pb.BenchGenerator('const', 10 ** (-6) + 10 ** (-12))),
    pb.BenchScalarArgument('cut_right', bm.precision.real_t,
                           pb.BenchGenerator('const', 10 ** (-6) - 10 ** (-12))),
)

size_list = [[30000000, 100000000], [1000, 2000]]
bench_histo = pb.PyBench(bf_histo, size_list, ['n_macro'])
bench_histo.bench(iterations=5, filename="histo/histo_{}_{}".format(args.precision, device),
                  gpu=gpu)
