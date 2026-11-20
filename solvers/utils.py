"""Timing and plot helpers shared by the experiments."""

import time

import matplotlib.pyplot as plt
import numpy as np


def timed(fn, Nruns_CPUtimes):
    """Run fn Nruns_CPUtimes times; return its last result and the median wall-clock time."""
    times = np.zeros(Nruns_CPUtimes)
    for rl in range(Nruns_CPUtimes):
        tic = time.perf_counter()
        result = fn()
        times[rl] = time.perf_counter() - tic
    return result, np.median(times)


def plot_convergence_history(hist, n_scenarios, tol, title):
    plt.figure()
    for res_vec, n in zip(hist, n_scenarios):
        plt.semilogy(res_vec, linewidth=2, label=f'n = {n}')
    plt.axhline(tol, color='k', linestyle='--', linewidth=2, label='Tolerance')
    plt.xlabel('Iteration number', fontsize=28)
    plt.ylabel('Relative residual', fontsize=28)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=28)
    plt.grid(True)
    plt.legend(fontsize=20, loc='upper right')


def plot_iterations_and_time(xdat1, ydat1, xlabel1, ylabel1, xdat2, ydat2, xlabel2, ylabel2):
    plt.figure()
    for sub, xdat, ydat, xlabel, ylabel in [(1, xdat1, ydat1, xlabel1, ylabel1), (2, xdat2, ydat2, xlabel2, ylabel2)]:
        plt.subplot(2, 1, sub)
        plt.plot(xdat, ydat, 'o-', linewidth=2, markersize=10, markerfacecolor='k', color='k')
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.xlim([0, xdat[-1]*1.05])
        plt.ylim([0, ydat[-1]*1.1])
        plt.grid(True)
    plt.tight_layout()
