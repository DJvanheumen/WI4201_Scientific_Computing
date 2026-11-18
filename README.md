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

### 4. Check the installation

```bash
python main.py
```

[main.py](main.py) is a sample script you can edit. It solves the Poisson problem on the unit square for $u = \sin(xy)$, prints the maximum error and plots the solution.

## MATLAB code

The original MATLAB scripts are in [Matlab code/](Matlab%20code/). To run them, open MATLAB in that folder and run `SCRIPT1_Discretization` (or one of the other scripts).