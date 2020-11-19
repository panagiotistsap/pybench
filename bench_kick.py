from pybench import *
import blond.utils.bmath as bm
import numpy as np

# gpu_kick(voltage, omega_rf, phi_rf, charge, n_rf, acceleration_kick, bm)
def f3():
    return 3


bench_kick = BenchFunction(
    bm.kick,
    BenchArrayArgument('dt', np.float64,
                       BenchGenerator('uniform', 0, 10), index=0),
    BenchArrayArgument('dE', np.float64,
                       BenchGenerator('uniform', 0, 10), index=0),
    BenchArrayArgument('voltage', np.float64,
                       BenchGenerator('uniform', 0, 10), index=1),
    BenchArrayArgument('omega_rf', np.float64,
                       BenchGenerator('uniform', 0, 10), index=1),
    BenchArrayArgument('phi_rf', np.float64,
                       BenchGenerator('uniform', 0, 10), index=1),
    BenchScalarArgument('charge', np.float64,
                        BenchGenerator('uniform', 0, 10)),
    BenchScalarArgument('n_rf', np.int32,
                        BenchGenerator('custom', generator_function=f3)),
    BenchScalarArgument('acceleration_kick', np.float64,
                        BenchGenerator('uniform', 0, 10)),
)

bench_kick.bench([[100000], [3]])
