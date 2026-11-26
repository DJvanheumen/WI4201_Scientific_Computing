"""One-level overlapping additive Schwarz preconditioner, eq. (5)-(8).

Named additive_schwarz rather than as, because `as` is a Python keyword and cannot be imported.
"""

import numpy as np
import scipy.sparse.linalg as spla

from tools.plotting import plot_cost_breakdown, plot_method_comparison, plot_subdomains_and_overlap
from tools.timing import timed

from .cg import CG_function, PCG_function
from .discretization import discretization_function
from .ssor import GaussSeidelSplitting_function, SSOR_preconditioner


def setup_AS_function(Ah, N, p_AS, l_AS):
    """Return the index sets of Omega_1..Omega_p and the factorizations of A_i = R_i Ah R_i^T.

    R_i is stored as its index set: R_i v = v[index_set[i]] and R_i^T v_i scatters v_i back.
    """

    # initialization
    index_set = [None] * p_AS
    CT_loc = [None] * p_AS

    # loop over subdomains
    for q_AS in range(1, p_AS + 1):

        # compute subdomain boundaries, eq. (6); the last strip owns column N+1 (eq. (5)), so every
        # unknown is covered also for l = 0
        i_min = max(1, N // p_AS * (q_AS-1) + 1 - l_AS)
        i_max = N+1 if q_AS == p_AS else min(N+1, N // p_AS * q_AS + l_AS)

        # global indices p = i + (N+1)(j-1) of all nodes with i_min <= i <= i_max, eq. (24)
        index_set[q_AS-1] = (np.arange(i_min-1, i_max)[:, None] + np.arange(N+1)[None, :] * (N+1)).reshape(-1, order='F')

        # local matrix A_i = R_i Ah R_i^T, eq. (7)
        A_loc = Ah[index_set[q_AS-1]][:, index_set[q_AS-1]].tocsc()

        # sparse factorization (symmetric ordering, as A_i is SPD)
        CT_loc[q_AS-1] = spla.splu(A_loc, permc_spec='MMD_AT_PLUS_A')

    return index_set, CT_loc


def AS_preconditioner(index_set, CT_loc):
    """Return r -> M_AS^{-1} r = sum_i R_i^T A_i^{-1} R_i r, eq. (8)."""

    def Additive_Schwarz_function(r):
        z = np.zeros_like(r)
        for idx, CT in zip(index_set, CT_loc):
            # restriction, local solve and prolongation, eq. (82)-(85)
            z[idx] += CT.solve(r[idx])
        return z

    return Additive_Schwarz_function


def run(N, p_scenarios, l_scenarios, p_compare, l_compare, omega, tol, maxit, Nruns_CPUtimes, figures_dir):
    """Implementation 8: AS-PCG iterations, setup time and PCG time for various p and l (p should divide N).

    For 8(d), CG, SSOR-PCG and AS-PCG with p_compare and l_compare are compared at the same N.
    """
    print("\n4. AS-PCG")

    h, n, u_ref, fh, Ah, u0 = discretization_function(N)

    # initialization
    CPUtimes_AS_phase1 = np.zeros((len(p_scenarios), len(l_scenarios)))
    CPUtimes_AS_phase2 = np.zeros((len(p_scenarios), len(l_scenarios)))
    NumberIterations_AS = np.zeros((len(p_scenarios), len(l_scenarios)), dtype=int)

    # loop over scenarios
    for pl, p_AS in enumerate(p_scenarios):
        for ll, l_AS in enumerate(l_scenarios):

            # set-up phase
            (index_set, CT_loc), CPUtimes_AS_phase1[pl, ll] = timed(lambda: setup_AS_function(Ah, N, p_AS, l_AS), Nruns_CPUtimes)

            # AS-PCG algorithm
            M_AS_inv = AS_preconditioner(index_set, CT_loc)
            (u_AS, k_AS, res_vec_AS), CPUtimes_AS_phase2[pl, ll] = timed(
                lambda: PCG_function(Ah, fh, u0, tol, maxit, M_AS_inv), Nruns_CPUtimes)
            NumberIterations_AS[pl, ll] = k_AS

    print("\nNumberIterations_AS:")
    print(NumberIterations_AS)
    print("\nCPUtimes_AS_phase1:")
    print(CPUtimes_AS_phase1)
    print("\nCPUtimes_AS_phase2:")
    print(CPUtimes_AS_phase2)

    # figure (not required by the assignment; supports questions 8(a)-(c))
    plot_subdomains_and_overlap(l_scenarios, p_scenarios, NumberIterations_AS,
                                CPUtimes_AS_phase1, CPUtimes_AS_phase2, N, figures_dir)

    # 8(d): CG, SSOR-PCG and AS-PCG at the same N (setup excluded from the iteration times, timed separately)
    def setup_SSOR():
        D, _, _, M1, M2, _, _ = GaussSeidelSplitting_function(Ah, n, omega)
        return SSOR_preconditioner(D, M1, M2, omega)

    M_SSOR_inv, setup_time_SSOR = timed(setup_SSOR, Nruns_CPUtimes)
    (index_set, CT_loc), setup_time_AS = timed(lambda: setup_AS_function(Ah, N, p_compare, l_compare), Nruns_CPUtimes)
    M_AS_inv = AS_preconditioner(index_set, CT_loc)
    setup_times = {'CG': 0.0, 'SSOR-PCG': setup_time_SSOR, 'AS-PCG': setup_time_AS}

    method_runs = {
        'CG': timed(lambda: CG_function(Ah, fh, u0, tol, maxit), Nruns_CPUtimes),
        'SSOR-PCG': timed(lambda: PCG_function(Ah, fh, u0, tol, maxit, M_SSOR_inv), Nruns_CPUtimes),
        'AS-PCG': timed(lambda: PCG_function(Ah, fh, u0, tol, maxit, M_AS_inv), Nruns_CPUtimes),
    }

    print(f"\nComparison at N = {N} (AS-PCG: p = {p_compare}, l = {l_compare}):")
    for method, ((u_m, k_m, res_m), time_m) in method_runs.items():
        print(f"  {method:9s} K = {k_m:5d}   iteration time = {time_m:.4f} s   setup time = {setup_times[method]:.4f} s"
              f"   total = {time_m + setup_times[method]:.4f} s")

    # AS-PCG residual histories for p_compare and every overlap l
    overlap_hist = []
    for l_AS in l_scenarios:
        u_AS, k_AS, res_vec_AS = PCG_function(Ah, fh, u0, tol, maxit, AS_preconditioner(*setup_AS_function(Ah, N, p_compare, l_AS)))
        overlap_hist.append(res_vec_AS)

    method_hist = {method: res_m for method, ((u_m, k_m, res_m), time_m) in method_runs.items()}
    plot_method_comparison(method_hist, overlap_hist, l_scenarios, p_compare, l_compare, tol, N, figures_dir)

    # extra figure for the presentation (not required): setup and iteration time per method
    plot_cost_breakdown(list(method_runs), list(setup_times.values()),
                        [time_m for (_, time_m) in method_runs.values()],
                        [k_m for ((_, k_m, _), _) in method_runs.values()],
                        N, p_compare, l_compare, figures_dir)


def run_scaling(N_scenarios, p_AS, l_AS, tol, maxit, Nruns_CPUtimes):
    """AS-PCG with fixed p and l for every N (p should divide N, l <= N/p), for the time scaling figure.

    Not required by the assignment. Returns {'AS-PCG': {'K': ..., 'time': ..., 'setup': ...}} per N.
    """
    K = np.zeros(len(N_scenarios), dtype=int)
    iteration_times, setup_times = np.zeros(len(N_scenarios)), np.zeros(len(N_scenarios))

    for nl, N in enumerate(N_scenarios):
        h, n, u_ref, fh, Ah, u0 = discretization_function(N)
        (index_set, CT_loc), setup_times[nl] = timed(lambda: setup_AS_function(Ah, N, p_AS, l_AS), Nruns_CPUtimes)
        M_AS_inv = AS_preconditioner(index_set, CT_loc)
        (u_AS, K[nl], res_vec_AS), iteration_times[nl] = timed(
            lambda: PCG_function(Ah, fh, u0, tol, maxit, M_AS_inv), Nruns_CPUtimes)

    return {'AS-PCG': {'K': K, 'time': iteration_times, 'setup': setup_times}}


if __name__ == "__main__":
    run(N=200, p_scenarios=[2, 4, 8, 10, 20], l_scenarios=[0, 1, 2, 4, 10], p_compare=4, l_compare=10, omega=1.5,
        tol=1e-10, maxit=20000, Nruns_CPUtimes=1, figures_dir="figures")
