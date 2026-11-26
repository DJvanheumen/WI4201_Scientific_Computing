"""WI4201 Scientific Computing, Assignment 2.

Solves the Poisson problem -Laplace(u) = f on the unit square, with exact solution u = sin(xy),
using a sparse direct solver, SSOR, CG, SSOR-preconditioned CG and additive Schwarz (AS)
preconditioned CG. The results are printed and the figures are saved as PNG in figures/.

Usage:  python main.py

All parameters are set below. To run one part on its own, use e.g. python -m solvers.ssor
(that uses the values in the __main__ block at the bottom of the module).
"""

import os

# Single-threaded BLAS, set before numpy is imported: the implementation is serial, and for these vector
# sizes thread overhead made CG up to 6x slower and the CPU times erratic.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from solvers import additive_schwarz, cg, discretization, ssor  # noqa: E402
from tools.plotting import plot_time_scaling  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent / "figures"

# ----------------------------------------------------------------------------
# Settings shared by all iterative methods
# ----------------------------------------------------------------------------
tol = 1e-10            # stopping criterion ||r^k||_2 / ||f^h||_2 <= tol, eq. (4)
maxit = 20000          # maximum number of iterations (SSOR needs 11320 at N = 200)
omega = 1.5            # SSOR relaxation parameter
Nruns_CPUtimes = 1     # CPU time is the median over this many runs

# ----------------------------------------------------------------------------
# Implementation 2: discretization error
# ----------------------------------------------------------------------------
q_scenarios = np.arange(2, 9)              # N = 2^q

# ----------------------------------------------------------------------------
# Implementation 3-6: SSOR, CG and SSOR-PCG
# ----------------------------------------------------------------------------
N_scenarios = np.array([40, 80, 120, 160, 200])   # number of subintervals per direction
Nasympt = 5                                # SSOR: number of final iterations for the residual ratios

# ----------------------------------------------------------------------------
# Implementation 8: AS-PCG
# ----------------------------------------------------------------------------
N_AS = N_scenarios[-1]                     # mesh size, p should divide N_AS
p_scenarios = [2, 4, 8, 10, 20]            # number of subdomains
l_scenarios = [0, 1, 2, 4, 10]             # overlap (grid layers), l = 0 is required by the assignment
p_compare, l_compare = 4, 10               # AS-PCG settings for the comparison with CG and SSOR-PCG, 8(d)


def main():
    tic = time.perf_counter()

    discretization.run(q_scenarios, FIGURES_DIR)
    results_SSOR = ssor.run(N_scenarios, omega, tol, maxit, Nasympt, Nruns_CPUtimes, FIGURES_DIR)
    results_CG = cg.run(N_scenarios, omega, tol, maxit, Nruns_CPUtimes, FIGURES_DIR)
    additive_schwarz.run(N_AS, p_scenarios, l_scenarios, p_compare, l_compare, omega, tol, maxit, Nruns_CPUtimes,
                         FIGURES_DIR)

    # Extra for the presentation (not required): total time (setup + iterations) of all methods against n
    print("\n5. TIME SCALING (setup + iterations)")
    results_AS = additive_schwarz.run_scaling(N_scenarios, p_compare, l_compare, tol, maxit, Nruns_CPUtimes)
    results = {**results_SSOR, **results_CG, **results_AS}
    n_scenarios = (N_scenarios+1)**2

    print("\n" + " " * 10 + "".join(f"{f'n = {n}':>12s}" for n in n_scenarios))
    for method, res in results.items():
        print(f"{method:10s}" + "".join(f"{t:11.3f}s" for t in res["setup"] + res["time"]))

    plot_time_scaling(n_scenarios, results, FIGURES_DIR)

    print(f"\nDone in {time.perf_counter() - tic:.1f} s. Figures are saved in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
