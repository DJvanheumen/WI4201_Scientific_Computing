"""One-level overlapping additive Schwarz preconditioner, eq. (5)-(8).

Named additive_schwarz rather than as, because `as` is a Python keyword and cannot be imported.
"""

import numpy as np
import scipy.sparse.linalg as spla

from .cg import PCG_function
from .discretization import discretization_function
from .utils import timed


def setup_AS_function(Ah, N, p_AS, l_AS):
    """Return the index sets of Omega_1..Omega_p and the factorizations of A_i = R_i Ah R_i^T.

    R_i is stored as its index set: R_i v = v[index_set[i]] and R_i^T v_i scatters v_i back.
    """

    # initialization
    index_set = [None] * p_AS
    CT_loc = [None] * p_AS

    # loop over subdomains
    for q_AS in range(1, p_AS + 1):

        # compute subdomain boundaries, eq. (6)
        i_min = max(1, N // p_AS * (q_AS-1) + 1 - l_AS)
        i_max = min(N+1, N // p_AS * q_AS + l_AS)

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


def run(N, p_scenarios, l_scenarios, tol, maxit, Nruns_CPUtimes):
    """Implementation 8: AS-PCG iterations, setup time and PCG time for various p and l (p should divide N)."""
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


if __name__ == "__main__":
    run(N=80, p_scenarios=[2, 4, 8, 10], l_scenarios=[1, 2, 4, 8], tol=1e-10, maxit=10000, Nruns_CPUtimes=1)
