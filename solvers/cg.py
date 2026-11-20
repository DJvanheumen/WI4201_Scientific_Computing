"""Conjugate gradient (eq. 38-42) and preconditioned conjugate gradient (eq. 55-60)."""

import matplotlib.pyplot as plt
import numpy as np

from .discretization import discretization_function
from .ssor import GaussSeidelSplitting_function, SSOR_preconditioner
from .utils import plot_convergence_history, plot_iterations_and_time, timed


def PCG_function(Ah, fh, u0, tol, maxit, M_inv):
    """Run PCG with preconditioner r -> M_inv(r); return u, k and the relative residual history."""

    # initialization
    u = u0.copy()
    r = fh - Ah @ u
    norm_fh = np.linalg.norm(fh)
    res_vec = [np.linalg.norm(r) / norm_fh]

    # p^0 = 0 makes the first search direction p^1 = z^0
    p = np.zeros_like(r)
    rTz_old = 1.0
    k = 0

    while res_vec[-1] > tol and k < maxit:

        # iteration counter
        k += 1

        # preconditioning, eq. (55)
        z = M_inv(r)
        rTz_new = r @ z

        # search direction, eq. (56)-(57)
        beta = rTz_new / rTz_old
        p = z + beta * p
        rTz_old = rTz_new

        # updates, eq. (58)-(60)
        Ap = Ah @ p
        alpha = rTz_old / (p @ Ap)
        u += alpha * p
        r -= alpha * Ap

        res_vec.append(np.linalg.norm(r) / norm_fh)

    return u, k, np.array(res_vec)


def CG_function(Ah, fh, u0, tol, maxit):
    """Run CG without preconditioning; PCG with M_inv = I reduces to eq. (38)-(42)."""
    return PCG_function(Ah, fh, u0, tol, maxit, lambda r: r)


def run(N_scenarios, omega, tol, maxit, Nruns_CPUtimes):
    """Implementation 6: convergence history and computational cost of CG and SSOR-PCG."""
    print("\n3. CG AND SSOR-PCG")

    n_scenarios = (N_scenarios+1)**2

    # preprocessing
    relres_CG_hist, relres_PCG_hist = [], []
    CPUtimes_CG, CPUtimes_PCG = np.zeros(len(N_scenarios)), np.zeros(len(N_scenarios))
    NumberIterations_CG, NumberIterations_PCG = np.zeros(len(N_scenarios), dtype=int), np.zeros(len(N_scenarios), dtype=int)

    # loop over scenarios
    for nl, N in enumerate(N_scenarios):
        h, n, u_ref, fh, Ah, u0 = discretization_function(N)
        D, _, _, M1, M2, _, _ = GaussSeidelSplitting_function(Ah, n, omega)

        # CG
        (u_CG, k_CG, res_vec_CG), CPUtimes_CG[nl] = timed(lambda: CG_function(Ah, fh, u0, tol, maxit), Nruns_CPUtimes)
        NumberIterations_CG[nl] = k_CG
        relres_CG_hist.append(res_vec_CG)

        # SSOR-PCG (preconditioner setup included in the timing)
        (u_PCG, k_PCG, res_vec_PCG), CPUtimes_PCG[nl] = timed(
            lambda: PCG_function(Ah, fh, u0, tol, maxit, SSOR_preconditioner(D, M1, M2, omega)), Nruns_CPUtimes)
        NumberIterations_PCG[nl] = k_PCG
        relres_PCG_hist.append(res_vec_PCG)

    # plots and output
    xdat1 = np.sqrt(n_scenarios) * np.log(n_scenarios)
    xdat2 = n_scenarios**(3/2) * np.log(n_scenarios)

    plot_convergence_history(relres_CG_hist, n_scenarios, tol, 'CG convergence history')
    plot_iterations_and_time(xdat1, NumberIterations_CG, 'n^(1/2) log(n)', '# CG iterations',
                             xdat2, CPUtimes_CG, 'n^(3/2) log(n)', 'CG CPU time (s)')

    plot_convergence_history(relres_PCG_hist, n_scenarios, tol, 'SSOR-PCG convergence history')
    plot_iterations_and_time(xdat1, NumberIterations_PCG, 'n^(1/2) log(n)', '# SSOR-PCG iterations',
                             xdat2, CPUtimes_PCG, 'n^(3/2) log(n)', 'SSOR-PCG CPU time (s)')

    print("\nNumberIterations_CG:")
    print(NumberIterations_CG)
    print("\nCPUtimes_CG:")
    print(CPUtimes_CG)
    print("\nNumberIterations_PCG:")
    print(NumberIterations_PCG)
    print("\nCPUtimes_PCG:")
    print(CPUtimes_PCG)


if __name__ == "__main__":
    run(N_scenarios=np.array([20, 40, 60, 80]), omega=1.5, tol=1e-10, maxit=10000, Nruns_CPUtimes=1)
    plt.show()
