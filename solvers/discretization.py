"""Five-point discretization of -Laplace(u) = f with u_ex = sin(xy), boundary nodes kept as unknowns."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tools.plotting import plot_discretization_error


def discretization_function(N):
    """Return h, n, u_ref, fh, Ah, u0 for N subintervals per direction (x-lexicographic ordering)."""

    # discretization
    h = 1 / N
    n = (N + 1)**2

    # grid coordinates, eq. (22)-(23)
    x = np.arange(N + 1) * h
    y = np.arange(N + 1) * h
    X, Y = np.meshgrid(x, y, indexing='ij')

    # reference field, eq. (9)
    u_ref = np.sin(X * Y)

    # source function, eq. (20)
    f_src = (X**2 + Y**2) * np.sin(X * Y)

    # boundary data, eq. (21)
    b_bdry = np.zeros((N + 1, N + 1))
    b_bdry[N, :] = np.sin(y)
    b_bdry[:, N] = np.sin(x)

    # construct fh, eq. (18)-(19)
    f_bar = f_src.copy()
    f_bar[1, :] += b_bdry[0, :] / h**2
    f_bar[N-1, :] += b_bdry[N, :] / h**2
    f_bar[:, 1] += b_bdry[:, 0] / h**2
    f_bar[:, N-1] += b_bdry[:, N] / h**2

    fh = b_bdry.copy()
    fh[1:N, 1:N] = f_bar[1:N, 1:N]
    fh = fh.reshape(-1, order='F')

    # construct Th_hat, eq. (16)-(17)
    Th = sp.diags([-1.0, 4.0, -1.0], offsets=[-1, 0, 1], shape=(N-1, N-1))
    Th_hat = sp.block_diag([[[h**2]], Th, [[h**2]]], format='csr')

    # construct J_hat, eq. (68); I_hat of eq. (15) is the same matrix
    J_diag = np.ones(N+1)
    J_diag[[0, N]] = 0
    J_hat = sp.diags(J_diag, format='csr')
    Ih_hat = J_hat

    # construct H_hat, eq. (69)
    H_offdiag = np.ones(N)
    H_offdiag[[0, N-1]] = 0
    H_hat = sp.diags([H_offdiag, H_offdiag], offsets=[-1, 1], format='csr')

    # construct Ah, eq. (67)
    I = sp.eye(N+1)
    Ah = (sp.kron(I - J_hat, I) + sp.kron(J_hat, Th_hat) / h**2 - sp.kron(H_hat, Ih_hat) / h**2).tocsr()

    # initial guess
    u0 = np.zeros(n)

    return h, n, u_ref, fh, Ah, u0


def run(q_scenarios, figures_dir):
    """Implementation 2: discretization error for N = 2^q, solved with a sparse direct solver."""
    print("\n1. Discretization")

    error_scenarios = np.zeros(len(q_scenarios))

    for il, q in enumerate(q_scenarios):
        N = 2**q
        h, n, u_ref, fh, Ah, _ = discretization_function(N)

        # direct solver
        u_h = spla.splu(Ah.tocsc()).solve(fh).reshape((N+1, N+1), order='F')
        error_scenarios[il] = np.max(np.abs(u_h - u_ref))

    # estimate order of convergence, eq. (73) with h_A the finest mesh
    xvals = 2.0**(-q_scenarios)
    yvals = error_scenarios
    beta_estimates = np.log(yvals[-1] / yvals[:-1]) / np.log(xvals[-1] / xvals[:-1])

    print("\nbeta_estimates:")
    print(beta_estimates)

    plot_discretization_error(xvals, yvals, figures_dir)


if __name__ == "__main__":
    run(q_scenarios=np.arange(2, 9), figures_dir="figures")
