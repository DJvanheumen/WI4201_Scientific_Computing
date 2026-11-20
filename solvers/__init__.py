"""Iterative solvers for the 2D Poisson problem -Laplace(u) = f on the unit square.

Each module has a run() for its part of the report, e.g. python -m solvers.ssor:
    utils              timing and plot helpers
    discretization    model problem, Ah and fh (Implementation 1-2)
    ssor               SSOR iteration and SSOR preconditioner (Implementation 3-5)
    cg                 CG and PCG (Implementation 6)
    additive_schwarz   one-level additive Schwarz preconditioner (Implementation 7-8)
"""
