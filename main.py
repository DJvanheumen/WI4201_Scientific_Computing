"""Sample script: solve -Laplace(u) = f on the unit square with a 5-point stencil.

Reference solution u = sin(xy), as in Matlab code/SCRIPT1_Discretization.m.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def laplacian_2d(N):
    """Return the 2D Laplacian on the (N-1)^2 interior grid points."""
    h = 1 / N
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(N - 1, N - 1)) / h**2
    I = sp.identity(N - 1)
    return (sp.kron(I, T) + sp.kron(T, I)).tocsr()


def main():
    N = 32
    h = 1 / N
    x = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")

    u_ref = np.sin(X * Y)
    f = (X**2 + Y**2) * np.sin(X * Y)

    # Move the known boundary values to the right-hand side.
    rhs = f[1:-1, 1:-1].copy()
    rhs[0, :] += u_ref[0, 1:-1] / h**2
    rhs[-1, :] += u_ref[-1, 1:-1] / h**2
    rhs[:, 0] += u_ref[1:-1, 0] / h**2
    rhs[:, -1] += u_ref[1:-1, -1] / h**2

    A = laplacian_2d(N)
    u = u_ref.copy()
    u[1:-1, 1:-1] = spla.spsolve(A, rhs.ravel()).reshape(N - 1, N - 1)

    print(f"N = {N}, max error = {np.abs(u - u_ref).max():.3e}")

    plt.contourf(X, Y, u, levels=20)
    plt.colorbar(label="u")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Numerical solution")
    plt.show()


if __name__ == "__main__":
    main()
