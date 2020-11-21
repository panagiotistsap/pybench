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
    gpu = True
    bm.use_gpu()
    device = "gpu"
    profile_dtype = np.int32


def rfft_prod(size):
    return np.fft.rfft(np.histogram(np.random.normal(10 ** (-6), 10 ** (-12), 1000 * size), bins=size)[0])


bf_irfft = pb.BenchFunction(
    bm.irfft,
    pb.BenchArrayArgument('a', bm.precision.complex_t,
                          pb.BenchGenerator('custom', generator_function=rfft_prod, dynamic_args=['size'])),
    pb.BenchScalarArgument('n', np.int,
                           pb.BenchGenerator('custom', generator_function=lambda x: x, dynamic_args=['size'])),
    pb.BenchScalarArgument('caller_id', np.int,
                           pb.BenchGenerator('const', 1), key=True, enable=gpu)
)

size_list = [10000, 20000]
bench_irfft = pb.PyBench(bf_irfft, size_list, ['n_macro'])
bench_irfft.bench(iterations=5, filename="irfft/irfft_{}_{}".format(args.precision, device),
                  gpu=gpu, save=False)
bench_irfft.bench(iterations=5, filename="irfft/irfft_{}_{}".format(args.precision, device),
                  gpu=gpu)
