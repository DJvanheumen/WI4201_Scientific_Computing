"""Figures for the report: one function per figure type, each saved as a PNG in figures_dir.

Uses the non-interactive Agg backend, so running the code never opens a window.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 (the backend must be set before importing pyplot)
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.figsize": (6.0, 4.0),
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def save(fig, figures_dir, name):
    """Save fig as figures_dir/name.png and close it."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {figures_dir.name}/{path.name}")


def plot_discretization_error(h, error, figures_dir):
    """Discretization error against h on log-log axes, with an O(h^2) reference line."""
    fig, ax = plt.subplots()
    ax.loglog(h, error, "o-", color="k", label="direct solver")
    ax.loglog(h, error[-1] * (h / h[-1])**2, "--", color="gray", label=r"$\mathcal{O}(h^2)$")
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\|\mathbf{u}^h - \mathbf{u}^h_{\mathrm{ex}}\|_\infty$")
    ax.set_title("Discretization error")
    ax.legend()
    save(fig, figures_dir, "discretization_error")


def plot_convergence_history(hist, n_scenarios, tol, method, figures_dir, name):
    """Relative residual against iteration number, one line per problem size n."""
    fig, ax = plt.subplots()
    for res_vec, n in zip(hist, n_scenarios):
        ax.semilogy(np.arange(len(res_vec)), res_vec, label=f"$n = {n}$")
    ax.axhline(tol, color="k", linestyle="--", linewidth=1, label="tolerance")
    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"$\|\mathbf{r}^k\|_2 \, / \, \|\mathbf{f}^h\|_2$")
    ax.set_title(f"{method} convergence history")
    ax.legend()
    save(fig, figures_dir, name)


def plot_iterations_and_time(x_iter, iterations, x_iter_label, x_time, cpu_times, x_time_label,
                             method, figures_dir, name):
    """Iterations K and CPU time against the predicted scaling, each with a linear least-squares fit.

    A straight line confirms the predicted complexity, e.g. K = O(n) and CPU time = O(n^2) for SSOR.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    panels = [(x_iter, iterations, x_iter_label, f"{method} iterations $K$"),
              (x_time, cpu_times, x_time_label, f"{method} CPU time (s)")]

    for ax, (x, y, xlabel, ylabel) in zip(axes, panels):
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.array([0, 1.05 * x[-1]])
        ax.plot(x_fit, slope * x_fit + intercept, "--", color="gray", label="linear fit")
        ax.plot(x, y, "o", color="k", label="measured")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, x_fit[-1])
        ax.set_ylim(bottom=0)
        ax.legend()

    fig.tight_layout()
    save(fig, figures_dir, name)


def plot_method_comparison(method_hist, overlap_hist, l_scenarios, p_AS, l_AS, tol, N, figures_dir):
    """Left: CG, SSOR-PCG and AS-PCG at one problem size. Right: AS-PCG for each overlap l at fixed p.

    method_hist maps a method name to its relative residual history; overlap_hist has one history per l.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)

    for method, res_vec in method_hist.items():
        ax1.semilogy(np.arange(len(res_vec)), res_vec, label=method)
    ax1.set_title(f"Methods ($N = {N}$; AS: $p = {p_AS}$, $\\ell = {l_AS}$)")

    for l_val, res_vec in zip(l_scenarios, overlap_hist):
        ax2.semilogy(np.arange(len(res_vec)), res_vec, label=f"$\\ell = {l_val}$")
    ax2.set_title(f"AS-PCG overlap ($N = {N}$, $p = {p_AS}$)")

    for ax in (ax1, ax2):
        ax.axhline(tol, color="k", linestyle="--", linewidth=1, label="tolerance")
        ax.set_xlabel(r"iteration $k$")
        ax.legend()
    ax1.set_ylabel(r"$\|\mathbf{r}^k\|_2 \, / \, \|\mathbf{f}^h\|_2$")

    fig.tight_layout()
    save(fig, figures_dir, "method_comparison")


def plot_cost_breakdown(methods, setup_times, iteration_times, iterations, N, p_AS, l_AS, figures_dir):
    """Stacked bars of setup time and iteration time per method, with the iteration count K on top.

    Extra figure for the presentation: the assignment times only the iterations, which hides the setup cost.
    """
    fig, ax = plt.subplots()
    x = np.arange(len(methods))
    setup_times, iteration_times = np.asarray(setup_times), np.asarray(iteration_times)

    ax.bar(x, setup_times, width=0.6, color="tab:gray", label="setup")
    ax.bar(x, iteration_times, width=0.6, bottom=setup_times, color="tab:blue", label="iterations")
    for xi, total, k in zip(x, setup_times + iteration_times, iterations):
        ax.annotate(f"$K = {k}$", (xi, total), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    ax.set_xticks(x, methods)
    ax.set_ylabel("CPU time (s)")
    ax.set_ylim(0, 1.15 * (setup_times + iteration_times).max())
    ax.set_title(f"Setup and iteration time, $N = {N}$ (AS: $p = {p_AS}$, $\\ell = {l_AS}$)")
    ax.grid(axis="x", visible=False)
    ax.legend()
    save(fig, figures_dir, "cost_breakdown")


def plot_time_scaling(n_scenarios, results, figures_dir):
    """Total CPU time (setup + iterations) against n on log-log axes, one line per method.

    The legend gives the slope of a least-squares fit over the three largest n, i.e. the measured growth rate
    time ~ n^slope, to compare with the complexity estimates. Extra figure for the presentation.
    """
    fig, ax = plt.subplots()
    n = np.asarray(n_scenarios, dtype=float)

    for method, res in results.items():
        total = res["setup"] + res["time"]
        slope = np.polyfit(np.log(n[-3:]), np.log(total[-3:]), 1)[0]
        ax.loglog(n, total, "o-", label=f"{method} (slope {slope:.2f})")

    ax.set_xticks(n, [f"{int(n_i)}" for n_i in n])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"number of unknowns $n$")
    ax.set_ylabel("total CPU time, setup + iterations (s)")
    ax.set_title("Time to convergence against problem size")
    ax.legend()
    save(fig, figures_dir, "time_scaling")


def plot_subdomains_and_overlap(l_scenarios, p_scenarios, iterations, setup_times, pcg_times, N, figures_dir):
    """AS-PCG iterations, setup time and PCG time against the overlap l, one line per number of subdomains p."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    panels = [(iterations, "AS-PCG iterations $K$"),
              (setup_times, "setup time (s)"),
              (pcg_times, "PCG time (s)")]

    for ax, (data, ylabel) in zip(axes, panels):
        for p_AS, row in zip(p_scenarios, data):
            ax.plot(l_scenarios, row, "o-", label=f"$p = {p_AS}$")
        ax.set_xlabel(r"overlap $\ell$ (grid layers, $\delta = \ell h$)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(l_scenarios)
        ax.set_ylim(bottom=0)

    axes[0].legend()
    fig.suptitle(f"Additive Schwarz PCG, $N = {N}$, $h = 1/{N}$")
    fig.tight_layout()
    save(fig, figures_dir, "as_pcg_overlap")
