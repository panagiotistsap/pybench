import blond.utils.bmath as bm
import pybench.main as pb
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
    gpu = True
solver = "simple"
alpha_order = 0


def get_solver():
    global solver
    return solver.encode('utf-8')


def get_alpha_order():
    return alpha_order


bf_drift = pb.BenchFunction(
    bm.drift,
    pb.BenchArrayArgument('dt', bm.precision.real_t,
                          pb.BenchGenerator('normal', 10 ** (-6), 10 ** (-12)), index=0),
    pb.BenchArrayArgument('dE', bm.precision.real_t,
                          pb.BenchGenerator('normal', 10 ** 6, 10 ** 12), index=0),
    pb.BenchScalarArgument('solver_uft8', bytes,
                           pb.BenchGenerator('custom', generator_function=get_solver)),
    pb.BenchScalarArgument('t_rev', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 1)),
    pb.BenchScalarArgument('length_ratio', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 1)),
    pb.BenchScalarArgument('alpha_order', bm.precision.real_t,
                           pb.BenchGenerator('custom', generator_function=get_alpha_order)),
    pb.BenchScalarArgument('eta_0', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 1)),
    pb.BenchScalarArgument('eta_1', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 1)),
    pb.BenchScalarArgument('eta_2', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 1)),
    pb.BenchScalarArgument('alpha_0', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 1)),
    pb.BenchScalarArgument('alpha_1', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 1)),
    pb.BenchScalarArgument('alpha_2', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 1)),
    pb.BenchScalarArgument('beta', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 1)),
    pb.BenchScalarArgument('energy', bm.precision.real_t,
                           pb.BenchGenerator('normal', 0, 1)),
)

size_list = [20, 40, 80, 160]
for s in ['simple', 'exact']:
    solver = s
    bench_drift = pb.PyBench(bf_drift, size_list, ['n_macro'])
    bench_drift.bench(iterations=10, filename="drift/drift_{}_{}".format(s, device), gpu=gpu)

for s in ['legacy']:
    solver = s
    for a in [0, 1, 2]:
        alpha_order = a
        bench_drift = pb.PyBench(bf_drift, size_list, ['n_macro'])
        bench_drift.bench(iterations=5, filename="drift/drift_{}_{}_{}_{}".format(args.precision, s,
                                                                                  device, alpha_order), gpu=gpu)
