import blond.utils.bmath as bm
import pybench.main as pb
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Benchmarking Functions.')
parser.add_argument('--gpu', '-g', type=str, choices=['True', 'False'], default="False")
parser.add_argument('--precision', '-p', choices=['single', 'double'], default='double')
args = parser.parse_args()

device = "cpu"
gpu = False
bm.use_precision(args.precision)
if args.gpu == "True":
    bm.use_gpu()
    device = "gpu"
    gpu = True

n_rf = 3
bf_kick = pb.BenchFunction(
    bm.kick,
    pb.BenchArrayArgument('dt', bm.precision.real_t,
                          pb.BenchGenerator('normal', 10 ** (-6), 10 ** (-12)), index=0),
    pb.BenchArrayArgument('dE', bm.precision.real_t,
                          pb.BenchGenerator('normal', 10 ** 6, 10 ** 12), index=0),
    pb.BenchArrayArgument('voltage', bm.precision.real_t,
                          pb.BenchGenerator('normal', 0, 1), index=1),
    pb.BenchArrayArgument('omega_rf', bm.precision.real_t,
                          pb.BenchGenerator('normal', 0, 1), index=1),
    pb.BenchArrayArgument('phi_rf', bm.precision.real_t,
                          pb.BenchGenerator('normal', 0, 1), index=1),
    pb.BenchScalarArgument('charge', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 1)),
    pb.BenchScalarArgument('n_rf', np.int32,
                           pb.BenchGenerator('const', n_rf)),
    pb.BenchScalarArgument('acceleration_kick', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 10)),
)

input_sizes = [[2000, 4000, 8000, 16000], [n_rf, n_rf, n_rf, n_rf]]
bench_kick = pb.PyBench(bf_kick, input_sizes, ['n_macro', 'slices'])
bench_kick.bench(iterations=5, filename="kick/kick_{}_{}".format(args.precision,
                                                                 device), gpu=gpu)
