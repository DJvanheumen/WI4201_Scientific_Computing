import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

# ============================================================
# DISCRETIZATION FUNCTION
# ============================================================

def discretization_function(N):

    # discretization
    h = 1 / N
    n = (N + 1)**2

    # grid coordinates
    x = np.arange(N + 1) * h
    y = np.arange(N + 1) * h

    # coordinates
    X, Y = np.meshgrid(x, y, indexing='ij')

    # reference field
    u_ref = np.sin(X * Y)

    # source function
    f_src = (X**2 + Y**2) * np.sin(X * Y)

    # boundary data
    b_bdry = np.zeros((N + 1, N + 1))
    b_bdry[N, :] = np.sin(y)
    b_bdry[:, N] = np.sin(x)

    # construct fh
    f_bar = f_src.copy()
    f_bar[1, :] += b_bdry[0, :] / h**2
    f_bar[N-1, :] += b_bdry[N, :] / h**2
    f_bar[:, 1] += b_bdry[:, 0] / h**2
    f_bar[:, N-1] += b_bdry[:, N] / h**2

    fh = b_bdry.copy()
    fh[1:N, 1:N] = f_bar[1:N, 1:N]
    fh = fh.reshape(-1, order='F')

    # construct Th_hat
    Th = sp.diags([-np.ones(N-2), 4*np.ones(N-1), -np.ones(N-2)], offsets=[-1, 0, 1], shape=(N-1, N-1), format='csr')
    Th_hat = sp.lil_matrix((N+1, N+1))
    Th_hat[1:N, 1:N] = Th
    Th_hat[0, 0] = h**2
    Th_hat[N, N] = h**2
    Th_hat = Th_hat.tocsr()

    # construct I_hat
    Ih_hat = sp.lil_matrix((N+1, N+1))
    Ih_hat[1:N, 1:N] = sp.eye(N-1)
    Ih_hat = Ih_hat.tocsr()

    # construct main diagonal from Kronecker product
    J_diag = np.zeros(N+1)
    J_diag[1:N] = 1
    J_hat = sp.diags(J_diag, offsets=0,shape=(N+1, N+1),format='csr')

    # construct lower and upper diagonals
    H_hat = sp.diags([np.ones(N), np.ones(N)],offsets=[-1, 1],shape=(N+1, N+1),format='lil')
    H_hat[0, 1] = 0
    H_hat[1, 0] = 0
    H_hat[N, N-1] = 0
    H_hat[N-1, N] = 0
    H_hat = H_hat.tocsr()

    # construct Ah
    Ah = (sp.kron(sp.eye(N+1) - J_hat, sp.eye(N+1)) + sp.kron(J_hat, Th_hat) / h**2 - sp.kron(H_hat, Ih_hat) / h**2).tocsr()

    # initial guess
    u0 = np.zeros(n)

    return h, n, u_ref, fh, Ah, u0

# ============================================================
# GAUSS-SEIDEL SPLITTING FUNCTION
# ============================================================

def GaussSeidelSplitting_function(Ah, n, omega):

    # matrix splitting Ah = D - E - F
    D = sp.diags(Ah.diagonal(), 0, shape=(n, n), format='csr')
    E = -sp.tril(Ah, k=-1, format='csr')
    F = -sp.triu(Ah, k=1, format='csr')

    # construct SSOR matrices
    M1 = D / omega - E
    N1 = (1-omega) / omega * D + F

    M2 = D / omega - F
    N2 = (1-omega) / omega * D + E

    return D, E, F, M1, M2, N1, N2

# ============================================================
# SSOR FUNCTION
# ============================================================

def SSOR_function(Ah, M1, M2, N1, N2, fh, u0, tol, maxit):

    # initialization
    k = 0
    u = u0.copy()

    relres = np.linalg.norm(fh - Ah @ u, 2) / np.linalg.norm(fh, 2)
    relres_vec = []

    # factorize matrices once
    M1_solver = spla.splu(M1.tocsc())
    M2_solver = spla.splu(M2.tocsc())

    # main loop
    while relres > tol and k < maxit:

        # step 1: forward SOR sweep
        uh_half = M1_solver.solve(N1 @ u + fh)

        # step 2: backward SOR sweep
        uh_new = M2_solver.solve(N2 @ uh_half + fh)

        # update solution
        u = uh_new

        # compute relative residual
        relres = np.linalg.norm(fh - Ah @ u, 2) / np.linalg.norm(fh, 2)
        relres_vec.append(relres)

        # update iteration counter
        k += 1

    return u, k, np.array(relres_vec)

# ============================================================
# CG FUNCTION
# ============================================================

def CG_function(Ah, fh, u0, tol, maxit):

    # initialization
    u = u0.copy()
    r = fh - Ah @ u
    p = r.copy()

    rTr_old = r @ r
    rel_res = np.linalg.norm(r) / np.linalg.norm(fh)

    res_vec = np.zeros(maxit + 1)
    res_vec[0] = rel_res

    # initial iteration
    if rel_res > tol:
        Ap = Ah @ p
        alpha = rTr_old / (p @ Ap)
        u = u + alpha * p
        r = r - alpha * Ap
        rTr_new = r @ r
        rel_res = np.sqrt(rTr_new) / np.linalg.norm(fh)
        res_vec[1] = rel_res
        k = 1

    else:
        k = 0

    # more iterations
    while rel_res > tol and k < maxit:

        # iteration counter
        k += 1

        # updates
        beta = rTr_new / rTr_old
        p = r + beta * p
        rTr_old = rTr_new
        Ap = Ah @ p
        alpha = rTr_old / (p @ Ap)
        u = u + alpha * p
        r = r - alpha * Ap
        rTr_new = r @ r
        rel_res = np.sqrt(rTr_new) / np.linalg.norm(fh)
        res_vec[k] = rel_res

    # cut tail of residual vector
    res_vec = res_vec[:k+1]

    return u, k, res_vec

# ============================================================
# SSOR-PCG FUNCTION
# ============================================================

def PCG_SSOR_function(Ah, D, M1, M2, fh, u0, tol, maxit, omega):

    # initialization
    u = u0.copy()
    r = fh - Ah @ u
    rel_res = np.linalg.norm(r) / np.linalg.norm(fh)
    res_vec = np.zeros(maxit + 1)
    res_vec[0] = rel_res

    # factorize matrices once
    M1_solver = spla.splu(M1.tocsc())
    M2_solver = spla.splu(M2.tocsc())
    z = omega * (2-omega) * M2_solver.solve(D @ M1_solver.solve(r))
    p = z.copy()
    rTz_old = r @ z

    # initial iteration
    if rel_res > tol:
        Ap = Ah @ p
        alpha = rTz_old / (p @ Ap)
        u = u + alpha * p
        r = r - alpha * Ap
        rel_res = np.linalg.norm(r) / np.linalg.norm(fh)
        res_vec[1] = rel_res
        k = 1
    else:
        k = 0

    # more iterations
    while rel_res > tol and k < maxit:

        # iteration counter
        k += 1

        # updates
        z = omega * (2-omega) * M2_solver.solve(D @ M1_solver.solve(r))
        rTz_new = r @ z
        beta = rTz_new / rTz_old
        p = z + beta * p
        rTz_old = rTz_new
        Ap = Ah @ p
        alpha = rTz_old / (p @ Ap)
        u = u + alpha * p
        r = r - alpha * Ap
        rel_res = np.linalg.norm(r) / np.linalg.norm(fh)
        res_vec[k] = rel_res

    # cut tail of residual vector
    res_vec = res_vec[:k+1]

    return u, k, res_vec

# ============================================================
# SETUP ADDITIVE SCHWARZ FUNCTION
# ============================================================

def setup_AS_function(Ah, N, p_AS, l_AS):

    # initialization
    index_set = [None] * p_AS
    A_loc = [None] * p_AS
    CT_loc = [None] * p_AS
    R_loc = [None] * p_AS

    # loop over subdomains
    for q_AS in range(1, p_AS + 1):

        # compute subdomain boundaries
        i_min = max(1, int(N/p_AS*(q_AS-1) + 1 - l_AS))
        i_max = min(N+1, int(N/p_AS*q_AS + l_AS))

        # compute indices
        index_set[q_AS-1] = (np.arange(i_min-1, i_max)[:, None] + np.arange(N+1)[None, :] * (N+1)).reshape(-1, order='F')

        # construct sparse restriction matrix
        n_i = len(index_set[q_AS-1])
        R_loc[q_AS-1] = sp.csr_matrix((np.ones(n_i),(np.arange(n_i),index_set[q_AS-1])),shape=(n_i, (N+1)**2))

        # construct local matrix
        A_loc[q_AS-1] = (R_loc[q_AS-1] @ Ah @ R_loc[q_AS-1].T).tocsc()

        # sparse factorization
        CT_loc[q_AS-1] = spla.splu(A_loc[q_AS-1])

    return index_set, A_loc, CT_loc, R_loc

# ============================================================
# ADDITIVE SCHWARZ FUNCTION
# ============================================================

def Additive_Schwarz_function(r, R_loc, CT_loc, p_AS):

    # initialization
    z = np.zeros_like(r)

    # loop over subdomains
    for q_AS in range(p_AS):

        # restriction
        r_i = R_loc[q_AS] @ r

        # solve locally
        z_i = CT_loc[q_AS].solve(r_i)

        # prolongation
        z = z + R_loc[q_AS].T @ z_i

    return np.asarray(z).ravel()

# ============================================================
# AS-PCG FUNCTION
# ============================================================

def PCG_AS_function(Ah, R_loc, CT_loc, p_AS, fh, u0, tol, maxit):

    # initialization
    u = u0.copy()
    r = fh - Ah @ u
    rel_res = np.linalg.norm(r) / np.linalg.norm(fh)
    res_vec = np.zeros(maxit + 1)
    res_vec[0] = rel_res
    z = Additive_Schwarz_function(r, R_loc, CT_loc, p_AS)
    p = z.copy()
    rTz_old = r @ z

    # initial iteration
    if rel_res > tol:
        Ap = Ah @ p
        alpha = rTz_old / (p @ Ap)
        u = u + alpha * p
        r = r - alpha * Ap
        rel_res = np.linalg.norm(r) / np.linalg.norm(fh)
        res_vec[1] = rel_res
        k = 1
    else:
        k = 0

    # more iterations
    while rel_res > tol and k < maxit:

        # iteration counter
        k += 1

        # updates
        z = Additive_Schwarz_function(r, R_loc, CT_loc, p_AS)
        rTz_new = r @ z
        beta = rTz_new / rTz_old
        p = z + beta * p
        rTz_old = rTz_new
        Ap = Ah @ p
        alpha = rTz_old / (p @ Ap)
        u = u + alpha * p
        r = r - alpha * Ap
        rel_res = np.linalg.norm(r) / np.linalg.norm(fh)
        res_vec[k] = rel_res

    # cut tail of residual vector
    res_vec = res_vec[:k+1]

    return u, k, res_vec

# ============================================================
# END OF FUNCTIONS
# ============================================================

# ============================================================
# MAIN PROGRAM
# ============================================================

# clearing and closing
plt.close('all')

# set chapters
chapters = [True, True, True]

# global parameter setting
tol = 1e-10
maxit = 10000
omega = 1.5
Nruns_CPUtimes = 1

# scenarios
q_scenarios = np.arange(2, 9) # q values in chapter 1
#N_scenarios = np.array([10, 20, 30, 40, 50])# N values in chapter 2
N_scenarios = np.array([20, 40, 60, 80])# N values in chapter 2
p_scenarios = [2, 4, 8, 10] # p values in chapter 3
l_scenarios = [1, 2, 4, 8] # l values in chapter 3
# l_scenarios = [0]  # scenarios without overlap in chapter 3 - takes relatively long as it does not converge

# ============================================================
# START CHAPTER 1: DISCRETIZATION
# ============================================================

if chapters[0] == True:
    print("\n")
    print("1. Discretization")

    # initialization
    error_scenarios = np.zeros(len(q_scenarios))

    # loop over scenarios
    for il in range(len(q_scenarios)):

        # parameter settings
        N = 2**q_scenarios[il]

        # discretization
        h, n, u_ref, fh, Ah, _ = discretization_function(N)

        # direct solver
        CT_Choleseky = spla.splu(Ah.tocsc())
        u_h = CT_Choleseky.solve(fh)
        u_h = u_h.reshape((N+1, N+1), order='F')

        # store errors
        error_scenarios[il] = np.max(np.abs(u_h - u_ref))

    # plot convergence
    xvals = 2.0**(-q_scenarios)
    yvals = error_scenarios

    plt.figure()
    plt.loglog(xvals, yvals, 'o-',linewidth=2, markersize=10,markerfacecolor='k', color='k')
    plt.xlabel(r'$h$', fontsize=32)
    plt.ylabel(r'$\|\mathbf{u}^h-\mathbf{u}^h_{\mathrm{ex}}\|_\infty$',fontsize=24)
    plt.title('Discretization error', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True)
    plt.box(True)

    # estimate order of convergence
    beta_estimates = (np.log(yvals[-1] / yvals[:-1]) / np.log(xvals[-1] / xvals[:-1]))

    # print output
    print("\nbeta_estimates:")
    print(beta_estimates)

# ============================================================
# END CHAPTER 1: DISCRETIZATION
# ============================================================

# ============================================================
# START CHAPTER 2: SSOR, CG AND SSOR-PCG
# ============================================================

if chapters[1] == True:
    print("\n")
    print("2. SSOR, CG AND SSOR-PCG")

    # compute n
    n_scenarios = (N_scenarios+1)**2

    # SSOR: preprocessing
    Nasympt = 5 # number of iterations for asymptotic analysis
    alpha_mat = np.zeros((len(N_scenarios), Nasympt))
    relres_hist = [None] * len(N_scenarios)
    CPUtimes_SSOR = np.zeros(len(N_scenarios))
    NumberIterations_SSOR = np.zeros(len(N_scenarios), dtype=int)

    # CG: preprocessing
    relres_CG_hist = [None] * len(N_scenarios)
    CPUtimes_CG = np.zeros(len(N_scenarios))
    NumberIterations_CG = np.zeros(len(N_scenarios), dtype=int)

    # SSOR-PCG: preprocessing
    relres_PCG_hist = [None] * len(N_scenarios)
    CPUtimes_PCG = np.zeros(len(N_scenarios))
    NumberIterations_PCG = np.zeros(len(N_scenarios), dtype=int)

    # loop over scenarios
    for nl in range(len(N_scenarios)):

        # parameter setting
        N = N_scenarios[nl]

        # discretization
        h, n, u_ref, fh, Ah, u0 = discretization_function(N)

        # Gauss-Seidel matrix splitting
        D, _, _, M1, M2, N1, N2 = GaussSeidelSplitting_function(Ah, n, omega)

        # SSOR: iterations
        times = np.zeros(Nruns_CPUtimes)
        for rl in range(Nruns_CPUtimes):
            tic = time.perf_counter()
            uh, k, relres_vec = SSOR_function(Ah, M1, M2, N1, N2, fh, u0, tol, maxit)
            times[rl] = time.perf_counter() - tic
        CPUtimes_SSOR[nl] = np.median(times)
        NumberIterations_SSOR[nl] = k
        relres_hist[nl] = relres_vec

        # CG: iterations
        times = np.zeros(Nruns_CPUtimes)
        for rl in range(Nruns_CPUtimes):
            tic = time.perf_counter()
            u_CG, k_CG, res_vec_CG = CG_function(Ah, fh, u0, tol, maxit)
            times[rl] = time.perf_counter() - tic
        CPUtimes_CG[nl] = np.median(times)
        NumberIterations_CG[nl] = k_CG
        u_CG = u_CG.reshape((N+1, N+1), order='F')
        relres_CG_hist[nl] = res_vec_CG

        # SSOR-CG: iterations
        times = np.zeros(Nruns_CPUtimes)
        for rl in range(Nruns_CPUtimes):
            tic = time.perf_counter()
            u_PCG, k_PCG, res_vec_PCG = PCG_SSOR_function(Ah, D, M1, M2, fh, u0, tol, maxit, omega)
            times[rl] = time.perf_counter() - tic
        CPUtimes_PCG[nl] = np.median(times)
        NumberIterations_PCG[nl] = k_PCG
        u_PCG = u_PCG.reshape((N+1, N+1), order='F')
        relres_PCG_hist[nl] = res_vec_PCG

    # SSOR: plot convergence history
    plt.figure()
    colors = ['r', 'b', 'g', 'm']
    for nl in range(len(N_scenarios)):
        plt.semilogy(np.arange(1, len(relres_hist[nl])+1),relres_hist[nl],color=colors[nl],linewidth=2,label=f'n = {n_scenarios[nl]}')
    plt.xlabel('Iteration number', fontsize=28)
    plt.ylabel('Relative residual', fontsize=28)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('SSOR convergence history', fontsize=28)
    plt.grid(True)
    plt.box(True)
    plt.axhline(tol, color='k', linestyle='--', linewidth=2)
    plt.legend(fontsize=20, loc='best')

    # SSOR: compute residual ratios for asymptotic convergence
    for il in range(len(n_scenarios)):
        for jl in range(Nasympt):
            alpha_mat[il, jl] = (relres_hist[il][-jl-1] / relres_hist[il][-jl-2])

    # SSOR: number of iterations and CPU times
    plt.figure()
    plt.subplot(2, 1, 1)  
    xdat1 = n_scenarios
    ydat1 = NumberIterations_SSOR
    plt.plot(xdat1, ydat1, 'o-', linewidth=2, markersize=10, markerfacecolor='k', color='k' )
    plt.xlabel('n', fontsize=16)
    plt.ylabel('# SSOR iterations', fontsize=16)
    plt.xlim([0, xdat1[-1]*1.05])
    plt.ylim([0, max(1, ydat1[-1]*1.1)])
    plt.grid(True)
    plt.subplot(2, 1, 2)
    xdat2 = n_scenarios**2
    ydat2 = CPUtimes_SSOR
    plt.plot(xdat2, ydat2, 'o-', linewidth=2, markersize=10, markerfacecolor='k', color='k')
    plt.xlabel('n^2', fontsize=16)
    plt.ylabel('SSOR CPU time (s)', fontsize=16)
    plt.xlim([0, xdat2[-1]*1.05])
    plt.ylim([0, max(1e-12, ydat2[-1]*1.1)])
    plt.grid(True)

    # SSOR: print output
    print("\nalpha_mat:")
    print(alpha_mat)
    print("\nNumberIterations_SSOR:")
    print(NumberIterations_SSOR)
    print("\nCPUtimes_SSOR:")
    print(CPUtimes_SSOR)

    # CG: plot convergence history
    plt.figure()
    colors = ['r', 'b', 'g', 'm', 'y']
    for nl in range(len(N_scenarios)):
        plt.semilogy(np.arange(len(relres_CG_hist[nl])),relres_CG_hist[nl],color=colors[nl],linewidth=2,label=f'n = {n_scenarios[nl]}')
    plt.xlabel('Iteration number', fontsize=28)
    plt.ylabel('Relative residual', fontsize=28)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('CG convergence history', fontsize=28)
    plt.grid(True)
    plt.box(True)
    plt.axhline(tol, color='k', linestyle='--', linewidth=2)
    plt.legend(fontsize=20, loc='upper right')

    # CG: plot number of iterations
    plt.figure()
    plt.subplot(2, 1, 1)
    xdat1 = np.sqrt(n_scenarios) * np.log(n_scenarios)
    ydat1 = NumberIterations_CG
    plt.plot(xdat1, ydat1, 'o-',linewidth=2, markersize=10,markerfacecolor='k', color='k')
    plt.xlabel('n^(1/2) log(n)', fontsize=16)
    plt.ylabel('# CG iterations', fontsize=16)
    plt.xlim([0, xdat1[-1]*1.05])
    plt.ylim([0, max(1, ydat1[-1]*1.1)])
    plt.grid(True)
    plt.subplot(2, 1, 2)
    xdat2 = n_scenarios**(3/2) * np.log(n_scenarios)
    ydat2 = CPUtimes_CG
    plt.plot(xdat2, ydat2, 'o-',linewidth=2, markersize=10,markerfacecolor='k', color='k')
    plt.xlabel('n^(3/2) log(n)', fontsize=16)
    plt.ylabel('CG CPU time (s)', fontsize=16)
    plt.xlim([0, xdat2[-1]*1.05])
    plt.ylim([0, max(1e-12, ydat2[-1]*1.1)])
    plt.grid(True)

    # CG: print output
    print("\nNumberIterations_CG:")
    print(NumberIterations_CG)
    print("\nCPUtimes_CG:")
    print(CPUtimes_CG)

    # SSOR-PCG: plot convergence history
    plt.figure()
    for nl in range(len(N_scenarios)):
        plt.semilogy(np.arange(len(relres_PCG_hist[nl])),relres_PCG_hist[nl],color=colors[nl],linewidth=2,label=f'n = {n_scenarios[nl]}')
    plt.xlabel('Iteration number', fontsize=28)
    plt.ylabel('Relative residual', fontsize=28)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('SSOR-PCG convergence history', fontsize=28)
    plt.grid(True)
    plt.box(True)
    plt.axhline(tol, color='k', linestyle='--', linewidth=2)
    plt.legend(fontsize=20, loc='upper right')

    # SSOR-PCG: plot number of iterations and computation time
    plt.figure()
    plt.subplot(2, 1, 1)
    xdat1 = np.sqrt(n_scenarios) * np.log(n_scenarios)
    ydat1 = NumberIterations_CG
    plt.plot(xdat1, ydat1, 'o-',linewidth=2, markersize=10,markerfacecolor='k', color='k')
    plt.xlabel('n^(1/2) log(n)', fontsize=16)
    plt.ylabel('# CG iterations', fontsize=16)
    plt.xlim([0, xdat1[-1]*1.05])
    plt.ylim([0, max(1, ydat1[-1]*1.1)])
    plt.grid(True)
    plt.subplot(2, 1, 2)
    xdat2 = n_scenarios**(3/2) * np.log(n_scenarios)
    ydat2 = CPUtimes_CG
    plt.plot(xdat2, ydat2, 'o-',linewidth=2, markersize=10,markerfacecolor='k', color='k')
    plt.xlabel('n^(3/2) log(n)', fontsize=16)
    plt.ylabel('CG CPU time (s)', fontsize=16)
    plt.xlim([0, xdat2[-1]*1.05])
    plt.ylim([0, max(1e-12, ydat2[-1]*1.1)])
    plt.grid(True)

    # SSOR-PCG: plot number of iterations and computation time
    plt.figure()
    plt.subplot(2, 1, 1)
    xdat1 = np.sqrt(n_scenarios) * np.log(n_scenarios)
    ydat1 = NumberIterations_PCG
    plt.plot(xdat1, ydat1, 'o-',linewidth=2, markersize=10,markerfacecolor='k', color='k')
    plt.xlabel('n^(1/2) log(n)', fontsize=16)
    plt.ylabel('# SSOR-PCG iterations', fontsize=16)
    plt.xlim([0, xdat1[-1]*1.05])
    plt.ylim([0, max(1, ydat1[-1]*1.1)])
    plt.grid(True)
    plt.subplot(2, 1, 2)
    xdat2 = n_scenarios**(3/2) * np.log(n_scenarios)
    ydat2 = CPUtimes_PCG
    plt.plot(xdat2, ydat2, 'o-', linewidth=2, markersize=10, markerfacecolor='k', color='k')
    plt.xlabel('n^(3/2) log(n)', fontsize=16)
    plt.ylabel('SSOR-PCG CPU time (s)', fontsize=16)
    plt.xlim([0, xdat2[-1]*1.05])
    plt.ylim([0, max(1e-12, ydat2[-1]*1.1)])
    plt.grid(True)

    # SSOR-PCG: print output
    print("\nNumberIterations_PCG:")
    print(NumberIterations_PCG)
    print("\nCPUtimes_PCG:")
    print(CPUtimes_PCG)

# ============================================================
# END CHAPTER 2: SSOR, CG AND SSOR-PCG
# ============================================================

# ============================================================
# START CHAPTER 3: AS-PCG
# ============================================================

if chapters[2] == True:
    print("\n")
    print("3. AS-PCG")

    # setting N
    N = N_scenarios[-1]

    # discretization
    h, n, u_ref, fh, Ah, u0 = discretization_function(N)

    # initialization
    CPUtimes_AS_phase1 = np.zeros((len(p_scenarios), len(l_scenarios)))
    CPUtimes_AS_phase2 = np.zeros((len(p_scenarios), len(l_scenarios)))
    NumberIterations_AS = np.zeros((len(p_scenarios), len(l_scenarios)),dtype=int)

    # loop over scenarios
    for pl in range(len(p_scenarios)):
        for ll in range(len(l_scenarios)):

            # parameter settings
            p_AS = p_scenarios[pl]
            l_AS = l_scenarios[ll]

            # set-up phase
            times = np.zeros(Nruns_CPUtimes)
            for rl in range(Nruns_CPUtimes):
                tic = time.perf_counter()
                index_set, A_loc, CT_loc, R_loc = setup_AS_function(Ah, N, p_AS, l_AS)
                times[rl] = time.perf_counter() - tic
            CPUtimes_AS_phase1[pl, ll] = np.median(times)

            # AS-PCG algorithm
            times = np.zeros(Nruns_CPUtimes)
            for rl in range(Nruns_CPUtimes):
                tic = time.perf_counter()
                u_AS, k_AS, res_vec_AS = PCG_AS_function(Ah, R_loc, CT_loc, p_AS, fh, u0, tol, maxit)
                times[rl] = time.perf_counter() - tic
            CPUtimes_AS_phase2[pl, ll] = np.median(times)
            NumberIterations_AS[pl, ll] = k_AS

    # print output
    print("\nNumberIterations_AS:")
    print(NumberIterations_AS)
    print("\nCPUtimes_AS_phase1:")
    print(CPUtimes_AS_phase1)
    print("\nCPUtimes_AS_phase2:")
    print(CPUtimes_AS_phase2)

# ============================================================
# END CHAPTER 3: AS-PCG
# ============================================================

# ============================================================
# SHOW ALL FIGURES
# ============================================================

plt.show()

