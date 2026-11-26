"""Wall-clock timing of the solvers."""

import time

import numpy as np


def timed(fn, Nruns_CPUtimes):
    """Run fn Nruns_CPUtimes times; return its last result and the median wall-clock time."""
    times = np.zeros(Nruns_CPUtimes)
    for rl in range(Nruns_CPUtimes):
        tic = time.perf_counter()
        result = fn()
        times[rl] = time.perf_counter() - tic
    return result, np.median(times)
