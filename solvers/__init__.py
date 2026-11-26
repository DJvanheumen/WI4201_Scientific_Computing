"""Iterative solvers for the 2D Poisson problem -Laplace(u) = f on the unit square.

Each module has a run() for its part of the report, e.g. python -m solvers.ssor:
    discretization     model problem, Ah and fh (Implementation 1-2)
    ssor               SSOR iteration and SSOR preconditioner (Implementation 3-5)
    cg                 CG and PCG (Implementation 6)
    additive_schwarz   one-level additive Schwarz preconditioner (Implementation 7-8)
"""

import os

# Single-threaded BLAS for python -m solvers.<module> (main.py sets the same before importing numpy).
# This package is imported before its modules import numpy, so the setting takes effect here as well.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
