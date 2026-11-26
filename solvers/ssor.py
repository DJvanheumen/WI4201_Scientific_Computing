"""SSOR iteration (eq. 33-34) and SSOR preconditioner (eq. 51)."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tools.plotting import plot_convergence_history, plot_iterations_and_time
from tools.timing import timed

from .discretization import discretization_function


def GaussSeidelSplitting_function(Ah, n, omega):
    """Return D, E, F, M1, M2, N1, N2 of the splitting Ah = D - E - F, eq. (27)-(32)."""

    # matrix splitting Ah = D - E - F
    D = sp.diags(Ah.diagonal(), 0, shape=(n, n), format='csr')
    E = -sp.tril(Ah, k=-1, format='csr')
    F = -sp.triu(Ah, k=1, format='csr')

    # construct SSOR matrices
    M1 = D / omega - E
    N1 = (1-omega) / omega * D + F

    M2 = D / omega - F
    N2 = (1-omega) / omega * D + E

    return D, E, F, M1, M2, N1, N2


def triangular_solver(M):
    """Return a solve function for triangular M (natural ordering keeps the factor equal to M, no fill-in)."""
    return spla.splu(M.tocsc(), permc_spec='NATURAL').solve


def SSOR_function(Ah, M1, M2, N1, N2, fh, u0, tol, maxit):
    """Run SSOR until ||r^k||_2 / ||fh||_2 <= tol; return u, k and the relative residual history."""

    # initialization
    k = 0
    u = u0.copy()
    norm_fh = np.linalg.norm(fh)

    relres = np.linalg.norm(fh - Ah @ u) / norm_fh
    relres_vec = [relres]

    # factorize matrices once
    M1_solve = triangular_solver(M1)
    M2_solve = triangular_solver(M2)

    # main loop
    while relres > tol and k < maxit:

        # step 1: forward SOR sweep, eq. (33)
        uh_half = M1_solve(N1 @ u + fh)

        # step 2: backward SOR sweep, eq. (34)
        u = M2_solve(N2 @ uh_half + fh)

        # compute relative residual
        relres = np.linalg.norm(fh - Ah @ u) / norm_fh
        relres_vec.append(relres)

        k += 1

    return u, k, np.array(relres_vec)


def SSOR_preconditioner(D, M1, M2, omega):
    """Return r -> M_SSOR^{-1} r = M2^{-1} ((2-omega)/omega D) M1^{-1} r, eq. (51)."""

    M1_solve = triangular_solver(M1)
    M2_solve = triangular_solver(M2)
    D_scaled = (2-omega) / omega * D.diagonal()

    return lambda r: M2_solve(D_scaled * M1_solve(r))


def run(N_scenarios, omega, tol, maxit, Nasympt, Nruns_CPUtimes, figures_dir):
    """Implementation 3-5: SSOR convergence history, asymptotic convergence and computational cost.

    Nasympt is the number of final iterations used for the residual ratios of eq. (74).
    Returns {'SSOR': {'K': ..., 'time': ..., 'setup': ...}} per N, for the time scaling figure in main.py.
    """
    print("\n2. SSOR")

    n_scenarios = (N_scenarios+1)**2

    # preprocessing
    alpha_mat = np.zeros((len(N_scenarios), Nasympt))
    relres_hist = []
    CPUtimes_SSOR = np.zeros(len(N_scenarios))
    setup_times_SSOR = np.zeros(len(N_scenarios))
    NumberIterations_SSOR = np.zeros(len(N_scenarios), dtype=int)

    # loop over scenarios
    for nl, N in enumerate(N_scenarios):
        h, n, u_ref, fh, Ah, u0 = discretization_function(N)
        (D, _, _, M1, M2, N1, N2), setup_times_SSOR[nl] = timed(
            lambda: GaussSeidelSplitting_function(Ah, n, omega), Nruns_CPUtimes)

        (uh, k, relres_vec), CPUtimes_SSOR[nl] = timed(lambda: SSOR_function(Ah, M1, M2, N1, N2, fh, u0, tol, maxit), Nruns_CPUtimes)
        NumberIterations_SSOR[nl] = k
        relres_hist.append(relres_vec)

    # residual ratios for asymptotic convergence, eq. (74)
    for il in range(len(n_scenarios)):
        for jl in range(Nasympt):
            alpha_mat[il, jl] = relres_hist[il][-jl-1] / relres_hist[il][-jl-2]

    # output
    print("\nalpha_mat:")
    print(alpha_mat)
    print("\nNumberIterations_SSOR:")
    print(NumberIterations_SSOR)
    print("\nCPUtimes_SSOR:")
    print(CPUtimes_SSOR)

    # figures: K = O(n) and CPU time = O(n^2), eq. (37)
    plot_convergence_history(relres_hist, n_scenarios, tol, 'SSOR', figures_dir, 'ssor_convergence')
    plot_iterations_and_time(n_scenarios, NumberIterations_SSOR, r'$n$',
                             n_scenarios**2, CPUtimes_SSOR, r'$n^2$',
                             'SSOR', figures_dir, 'ssor_cost')

    return {'SSOR': {'K': NumberIterations_SSOR, 'time': CPUtimes_SSOR, 'setup': setup_times_SSOR}}


if __name__ == "__main__":
    run(N_scenarios=np.array([40, 80, 120, 160, 200]), omega=1.5, tol=1e-10, maxit=20000, Nasympt=5, Nruns_CPUtimes=1,
        figures_dir="figures")
