"""WI4201 Assignment 2: run all experiments, print the results and show the figures.

Usage: python main.py
Run a single part with e.g. python -m solvers.ssor (uses the values in that module's __main__ block)
"""

import matplotlib.pyplot as plt
import numpy as np

from solvers import additive_schwarz, cg, discretization, ssor

# global parameter setting
tol = 1e-10            # stopping criterion ||r^k||_2 / ||f^h||_2 <= tol, eq. (4)
maxit = 10000          # maximum number of iterations
omega = 1.5            # SSOR relaxation parameter
Nruns_CPUtimes = 1     # CPU time is the median over this many runs

# scenarios
q_scenarios = np.arange(2, 9)              # discretization: N = 2^q
N_scenarios = np.array([20, 40, 60, 80])   # SSOR, CG and SSOR-PCG
Nasympt = 5                                # SSOR: number of final iterations for the residual ratios
N_AS = N_scenarios[-1]                     # AS-PCG: mesh size, p should divide N_AS
p_scenarios = [2, 4, 8, 10]                # AS-PCG: number of subdomains
l_scenarios = [1, 2, 4, 8]                 # AS-PCG: overlap
# l_scenarios = [0]  # scenarios without overlap - takes relatively long as it does not converge

# run all experiments
discretization.run(q_scenarios)
ssor.run(N_scenarios, omega, tol, maxit, Nasympt, Nruns_CPUtimes)
cg.run(N_scenarios, omega, tol, maxit, Nruns_CPUtimes)
additive_schwarz.run(N_AS, p_scenarios, l_scenarios, tol, maxit, Nruns_CPUtimes)

plt.show()
