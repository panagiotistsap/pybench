import blond.utils.bmath as bm
import pybench.main as pb
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Benchmarking Functions.')
parser.add_argument('--gpu', '-g', type=str, choices=['True', 'False'], default="False")
parser.add_argument('--precision', '-p', choices=['single', 'double'], default='double')
args = parser.parse_args()

device = "cpu"
bm.use_precision(args.precision)
gpu = False
if args.gpu == "True":
    bm.use_gpu()
    device = "gpu"
    gpu = False


def produce_bin_centers(slices, beam_dt):
    min_beam_dt = min(beam_dt)
    max_beam_dt = max(beam_dt)
    bin_width = (max_beam_dt - min_beam_dt) // slices
    bin_centers = np.linspace(min_beam_dt + bin_width // 2, max_beam_dt - bin_width // 2, slices)
    return bin_centers


n_rf = 3
bf_linear_interp_kick = pb.BenchFunction(
    bm.linear_interp_kick,
    pb.BenchArrayArgument('dt', bm.precision.real_t,
                          pb.BenchGenerator('normal', 10 ** (-6), 10 ** (-12)), index=0),
    pb.BenchArrayArgument('dE', bm.precision.real_t,
                          pb.BenchGenerator('normal', 10 ** 6, 10 ** 12), index=0),
    pb.BenchArrayArgument('voltage', bm.precision.real_t,
                          pb.BenchGenerator('normal', 0, 1), index=1),
    pb.BenchArrayArgument('bin_centers', bm.precision.real_t,
                          pb.BenchGenerator('custom', generator_function=produce_bin_centers,
                                            dynamic_args=['size', 'dt']), index=1),
    pb.BenchScalarArgument('charge', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 1)),
    pb.BenchScalarArgument('acceleration_kick', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 10)),
)

input_sizes = [[200000], [10]]
bench_linear_interp_kick = pb.PyBench(bf_linear_interp_kick, input_sizes, ['n_macro', 'slices'])
bench_linear_interp_kick.bench(iterations=5,
                               filename="linear_interp_kick/linear_interp_kick_{}_{}".
                               format(args.precision, device), gpu=gpu)
