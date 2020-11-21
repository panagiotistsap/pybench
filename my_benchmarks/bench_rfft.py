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
gpu = False
profile_dtype = np.float64
if args.gpu == "True":
    gpu = True
    bm.use_gpu()
    bm.enable_gpucache()
    device = "gpu"
    profile_dtype = np.int32


def hist_prod(size):
    return np.histogram(np.random.normal(10 ** (-6), 10 ** (-12), 1000 * size), bins=size)[0]


bf_rfft = pb.BenchFunction(
    bm.rfft,
    pb.BenchArrayArgument('a', profile_dtype,
                          pb.BenchGenerator('custom', generator_function=hist_prod, dynamic_args=['size'])),
    pb.BenchScalarArgument('n', np.int,
                           pb.BenchGenerator('custom', generator_function=lambda x: 10*x, dynamic_args=['size'])),
    pb.BenchScalarArgument('caller_id', np.int,
                           pb.BenchGenerator('const', 1), key=True, enable=gpu)
)

size_list = [10000]
bench_rfft = pb.PyBench(bf_rfft, size_list, ['n_macro'])
# Dummy to produce the plans
bench_rfft.bench(iterations=1, filename="rfft/rfft_{}_{}".format(args.precision, device),
                 gpu=gpu, save=False)
# real benchmark
bench_rfft.bench(iterations=10, filename="rfft/rfft_{}_{}".format(args.precision, device),
                 gpu=gpu)
