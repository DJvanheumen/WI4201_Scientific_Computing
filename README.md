# WI4201_Scientific_Computing
Here is the code that is related to the group assignment for [WI4201] Scientific Computing

## Installation

### Prerequisites

- Python 3.11 or newer
  - Windows / macOS: use the installer from [python.org](https://www.python.org/downloads/). On macOS, `brew install python` also works.
  - Debian / Ubuntu: the system Python leaves out `venv` and `pip`, so install them with `sudo apt install python3-venv python3-pip`
  - Fedora: `sudo dnf install python3`
- [Git](https://git-scm.com/downloads)
- Optional, a Git app so you can clone, commit, push and pull without the command line (see step 1):
  - [VS Code](https://code.visualstudio.com/) (Windows, macOS, Linux): Git is built into the **Source Control** panel. Its [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) also detects the `.venv` from step 2.
  - [GitHub Desktop](https://desktop.github.com/) (Windows, macOS)
  - [GitKraken](https://www.gitkraken.com/) (Windows, macOS, Linux)

### 1. Clone the repository

From the command line:

```bash
git clone https://github.com/DJvanheumen/WI4201_Scientific_Computing.git
cd WI4201_Scientific_Computing
```

Or with a Git app, using the URL `https://github.com/DJvanheumen/WI4201_Scientific_Computing.git`:

- **VS Code:** open the Command Palette (`Ctrl+Shift+P`, or `Cmd+Shift+P` on macOS), run **Git: Clone**, paste the URL and choose a folder. Click **Open** when VS Code asks.
- **GitHub Desktop:** go to **File → Clone repository → URL**, paste the URL, choose a local path and click **Clone**.
- **GitKraken:** go to **File → Clone Repo → Clone with URL**, choose where to clone, paste the URL and click **Clone the repo!**

After cloning, open a terminal in the repository folder for the next steps. In VS Code, use **Terminal → New Terminal**.

### 2. Create and activate a virtual environment

| OS | Command |
| --- | --- |
| Linux / macOS | `python3 -m venv .venv` |
| Windows | `python -m venv .venv` |

Activate it:

| OS | Command |
| --- | --- |
| Linux / macOS | `source .venv/bin/activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |

Once it's active, `(.venv)` appears at the start of your prompt, and `python` and `pip` run the environment's own copies on every OS. To leave the environment, run `deactivate`.

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `numpy`: arrays, grids and norms
- `scipy`: sparse matrices (`scipy.sparse`) and solvers (`scipy.sparse.linalg`, `scipy.linalg`)
- `matplotlib`: plots

If you add a package, add it to `requirements.txt` as well, so everyone gets the same environment.

### 4. Run the code

```bash
python main.py
```

This runs all experiments of the assignment in order and prints the results (iteration counts, CPU times, residual ratios, convergence orders). No windows open and no input is needed: the figures are saved as PNG in `figures/`, which is created if it does not exist.

With the default settings (problem sizes $N = 40, 80, 120, 160, 200$ and one timing run per measurement) it takes about one minute, two thirds of which is SSOR. Every timing run more adds roughly the same again, so 5 timing runs take about 4 minutes.

## Code structure

The code solves the Poisson problem $-\Delta u = f$ on the unit square with exact solution $u = \sin(xy)$, using the five-point stencil. It compares SSOR, CG, SSOR-preconditioned CG and CG with a one-level additive Schwarz (AS) preconditioner.

```
main.py                      parameters and the calls to each experiment
solvers/                     the numerical methods, one module per part of the assignment
  discretization.py          matrix Ah and right-hand side fh; discretization error
  ssor.py                    SSOR iteration and SSOR preconditioner
  cg.py                      CG and preconditioned CG
  additive_schwarz.py        additive Schwarz preconditioner
tools/                       helpers that are not part of the numerical methods
  plotting.py                figure style and figures, saved as PNG
  timing.py                  wall-clock timing of the solvers
figures/                     output: the figures, created by running the code
```

### Parameters

All parameters are set at the top of [main.py](main.py): the tolerance, the maximum number of iterations, $\omega$, the number of timing runs, the values of $N$, $p$ and $l$ for each experiment, and the subdomains $p$ and overlap $l$ used to compare AS-PCG with CG and SSOR-PCG. `main.py` passes them to the `run()` function of each module.

### Timing

CPU times are wall-clock times measured with Python's `time.perf_counter`, using the same routine for every method. Each time covers all iterations until the stopping criterion $\|r^k\|_2 / \|f^h\|_2 \le 10^{-10}$ is met, and excludes matrix assembly and preconditioner setup; the setup of the preconditioners is timed separately. With `Nruns_CPUtimes > 1`, each measurement is repeated that many times, one after the other, and the median is reported.

The code runs single-threaded: `main.py` limits the linear algebra library (BLAS) to one thread before NumPy is imported. For these vector sizes, multithreading made the timings slower and erratic.

### Figures

| File | Contents |
|---|---|
| `discretization_error.png` | error against $h$, with an $\mathcal{O}(h^2)$ reference line |
| `ssor_convergence.png`, `cg_convergence.png`, `ssor_pcg_convergence.png` | relative residual against iteration number, for each $n$ |
| `ssor_cost.png`, `cg_cost.png`, `ssor_pcg_cost.png` | iterations and CPU time against the predicted complexity, with a linear fit |
| `as_pcg_overlap.png` | AS-PCG iterations, setup time and PCG time against the overlap $l$, for each $p$ |
| `method_comparison.png` | residual histories of CG, SSOR-PCG and AS-PCG at one $N$, and of AS-PCG for each $l$ |
| `cost_breakdown.png` | setup and iteration time per method at one $N$ (extra, not required by the assignment) |
| `time_scaling.png` | total time (setup + iterations) of all methods against $n$ (extra, not required by the assignment) |

### Running one part

Each module in [solvers/](solvers/) has a `run()` function for its part of the assignment. To run only that part, use

```bash
python -m solvers.ssor
```

(or `solvers.discretization`, `solvers.cg`, `solvers.additive_schwarz`). This uses the parameter values in the `if __name__ == "__main__":` block at the bottom of that module, so you can change them while working on that part without touching `main.py`. The results for the report come from `main.py`.

## MATLAB code

The original MATLAB scripts are in [matlab/](matlab/). To run them, open MATLAB in that folder and run `SCRIPT1_Discretization` (or one of the other scripts).