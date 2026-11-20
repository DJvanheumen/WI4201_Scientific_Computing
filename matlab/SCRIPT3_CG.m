% clearing and closing
clc; close all; clear all; beep off;

% create scenarios
N_scenarios = [40,80,120,160,200,240];
n_scenarios = (N_scenarios+1).^2;

% parameter settings
tol = 1e-10;
maxit = 10000;
omega = 1.5;
Nruns_CPUtimes = 1;  % set to 7 to estimate CPU times

% initialization
relres_CG_hist = cell(1,length(N_scenarios));
relres_PCG_hist = cell(1,length(N_scenarios));
CPUtimes_CG = zeros(length(N_scenarios),1);
NumberIterations_CG = zeros(length(N_scenarios),1);
CPUtimes_PCG = zeros(length(N_scenarios),1);
NumberIterations_PCG = zeros(length(N_scenarios),1);

% loop over scenarions
for nl = 1:length(N_scenarios)

    % parameter settings
    N = N_scenarios(nl);
    h = 1/N;
    
    % initial guess
    u0 = zeros((N+1)^2,1);
    
    % grid coordinates
    x = (0:N)*h;
    y = (0:N)*h;
    
    % coordinates
    [X,Y] = ndgrid(x,y);
    
    % reference field
    u_ref = sin(X.*Y);
    
    % source function
    f_src = (X.^2 + Y.^2).*sin(X.*Y);
    
    % boundary data
    b_bdry = zeros(N+1,N+1);
    b_bdry(N+1,:) = sin(y);
    b_bdry(:,N+1) = sin(x);
    
    % construct f_h
    f_bar = f_src;
    f_bar(2,:) = f_bar(2,:) + b_bdry(1,:)/h^2;
    f_bar(N,:) = f_bar(N,:) + b_bdry(N+1,:)/h^2;
    f_bar(:,2) = f_bar(:,2) + b_bdry(:,1)/h^2;
    f_bar(:,N) = f_bar(:,N) + b_bdry(:,N+1)/h^2;
    f_h = b_bdry;
    f_h(2:N,2:N) = f_bar(2:N,2:N);
    clear f_bar;
    
    % construct Th_hat
    Th = spdiags(ones(N-1,1)*[-1 4 -1], [-1 0 1], N-1, N-1);
    Th_hat = sparse(N+1,N+1);
    Th_hat(2:N,2:N) = Th;
    Th_hat(1,1) = h^2;
    Th_hat(N+1,N+1) = h^2;
    clear Th;
    
    % construct I_hat
    Ih_hat = sparse(N+1,N+1);
    Ih_hat(2:N,2:N) = speye(N-1);
    
    % construct main diagonal from Kronecker product
    J_hat = spdiags([zeros(N+1,1) [0; ones(N-1,1); 0] zeros(N+1,1)], [ -1 0 1], N+1,N+1);
    
    % construct lower and upper diagonals for Kronecker product
    H_hat = spdiags(ones(N+1,1)*[1 1], [-1 1], N+1, N+1);
    H_hat(1,2) = 0; H_hat(2,1) = 0;
    H_hat(N+1,N) = 0; H_hat(N,N+1) = 0;
    
    % construct Ah
    Ah = kron(speye(N+1)-J_hat,speye(N+1)) + kron(J_hat,Th_hat) / h^2 - kron(H_hat,Ih_hat) / h^2;
    clear J_hat H_hat Th_hat Ih_hat;
    
    % matrix splitting Ah = D - E - F
    D = spdiags(diag(Ah), 0, n_scenarios(nl), n_scenarios(nl));
    E = -tril(Ah,-1);
    F = -triu(Ah,1);
    
    % construct SSOR matrices
    M1 = D/omega - E;
    M2 = D/omega - F;
    
    % apply CG without preconditionig
    times = zeros(Nruns_CPUtimes,1);
    for rl = 1:Nruns_CPUtimes
    tic;
    [u_CG, k_CG, res_vec_CG] = CG_function(Ah, f_h(:), u0, tol, maxit);
    times(rl) = toc;
    end
    CPUtimes_CG(nl) = median(times); 
    NumberIterations_CG(nl) = k_CG;
    u_CG = reshape(u_CG,N+1,N+1);
    
    % apply CG without with SSOR-conditionig
    times = zeros(Nruns_CPUtimes,1);
    for rl = 1:Nruns_CPUtimes
    tic;
    [u_PCG, k_PCG, res_vec_PCG] = PCG_SSOR_function(Ah, D, M1, M2, f_h(:), u0, tol, maxit, omega);
    times(rl) = toc;
    end
    CPUtimes_PCG(nl) = median(times); 
    NumberIterations_PCG(nl) = k_PCG;
    u_PCG = reshape(u_PCG,N+1,N+1);
    
    % store residuals
    relres_CG_hist{nl} = res_vec_CG;
    relres_PCG_hist{nl} = res_vec_PCG;

end

% plot CG convergence history
figure; semilogy(0:length(relres_CG_hist{1})-1, relres_CG_hist{1}, 'r-', 'LineWidth', 2);
hold on; semilogy(0:length(relres_CG_hist{2})-1, relres_CG_hist{2}, 'b-', 'LineWidth', 2);
hold on; semilogy(0:length(relres_CG_hist{3})-1, relres_CG_hist{3}, 'g-', 'LineWidth', 2);
hold on; semilogy(0:length(relres_CG_hist{4})-1, relres_CG_hist{4}, 'm-', 'LineWidth', 2);
hold on; semilogy(0:length(relres_CG_hist{5})-1, relres_CG_hist{5}, 'c-', 'LineWidth', 2);
xlabel('Iteration number', 'FontSize', 28);
ylabel('Relative residual', 'FontSize', 28);
set(gca, 'FontSize', 18, 'LineWidth', 1.2);
grid on; box on;
title('CG convergence history', 'FontSize', 28);
hold on;
yline(tol, 'k--', 'LineWidth', 2);
legend(sprintf('n = %d',n_scenarios(1)),sprintf('n = %d',n_scenarios(2)),sprintf('n = %d',n_scenarios(3)),sprintf('n = %d',n_scenarios(4)),sprintf('n = %d',n_scenarios(5)),'Tolerance','FontSize',20,'Location','northeast');

% plot PCG convergence history
figure; semilogy(0:length(relres_PCG_hist{1})-1, relres_PCG_hist{1}, 'r-', 'LineWidth', 2);
hold on; semilogy(0:length(relres_PCG_hist{2})-1, relres_PCG_hist{2}, 'b-', 'LineWidth', 2);
hold on; semilogy(0:length(relres_PCG_hist{3})-1, relres_PCG_hist{3}, 'g-', 'LineWidth', 2);
hold on; semilogy(0:length(relres_PCG_hist{4})-1, relres_PCG_hist{4}, 'm-', 'LineWidth', 2);
hold on; semilogy(0:length(relres_PCG_hist{5})-1, relres_PCG_hist{5}, 'c-', 'LineWidth', 2);
xlabel('Iteration number', 'FontSize', 28);
ylabel('Relative residual', 'FontSize', 28);
set(gca, 'FontSize', 18, 'LineWidth', 1.2);
grid on; box on;
title('SSOR-PCG convergence history', 'FontSize', 28);
hold on;
yline(tol, 'k--', 'LineWidth', 2);
legend(sprintf('n = %d',n_scenarios(1)),sprintf('n = %d',n_scenarios(2)),sprintf('n = %d',n_scenarios(3)),sprintf('n = %d',n_scenarios(4)),sprintf('n = %d',n_scenarios(5)),'Tolerance','FontSize',20,'Location','northeast');

% plot number of iterations and computation time for CG
figure;
subplot(2,1,1);
xdat1 = (n_scenarios(:)).^(1/2).*log(n_scenarios(:));
ydat1 = NumberIterations_CG(:);
plot(0, 0, '-'); hold on;
plot(xdat1, ydat1, 'o-','LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k','Color','k');
xlabel('n^{1/2} log(n)', 'FontSize', 16);
ylabel('# CG iterations', 'FontSize', 16);
set(gca, 'FontSize', 16);
xlim([0 xdat1(end)*1.05]);
ylim([0 ydat1(end)*1.1]);
grid on;
subplot(2,1,2);
xdat2 = (n_scenarios(:)).^(3/2) .* log(n_scenarios(:));
ydat2 = CPUtimes_CG(:);
plot(0, 0, '-'); hold on;
plot(xdat2 , ydat2, 'o-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k','Color','k');
xlabel('n^{2/3} log(n)', 'FontSize', 16);
ylabel('CG CPU time (s)', 'FontSize', 16);
set(gca, 'FontSize', 16);
xlim([0 xdat2(end)*1.05]);
ylim([0 ydat2(end)*1.1]);
grid on;

% plot number of iterations and computation time for PCG
figure;
subplot(2,1,1);
xdat1 = n_scenarios(:).^(1/2) .* log(n_scenarios(:));
ydat1 = NumberIterations_PCG(:);
plot(0, 0, '-'); hold on;
plot(xdat1, ydat1, 'o-','LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k','Color','k');
xlabel('n^{1/2} log(n)', 'FontSize', 16);
ylabel('# SSOR-PCG iterations', 'FontSize', 16);
set(gca, 'FontSize', 16);
xlim([0 xdat1(end)*1.05]);
ylim([0 ydat1(end)*1.1]);
grid on;
subplot(2,1,2);
xdat2 = (n_scenarios(:)).^(3/2) .* log(n_scenarios(:));
ydat2 = CPUtimes_PCG(:);
plot(0, 0, '-'); hold on;
plot(xdat2, ydat2, 'o-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k','Color','k');
xlabel('n^{3/2} log(n)', 'FontSize', 16);
ylabel('SSOR-PCG CPU time (s)', 'FontSize', 16);
set(gca, 'FontSize', 16);
xlim([0 xdat2(end)*1.05]);
ylim([0 ydat2(end)*1.1]);
grid on;

% print output
NumberIterations_CG
CPUtimes_CG
NumberIterations_PCG
CPUtimes_PCG

% CG function
function [u, k, res_vec] = CG_function(A, f, u0, tol, maxit)

    % initialization
    u = u0;
    r = f - A*u;
    p = r;
    rTr_old = r' * r;
    rel_res = norm(r)/norm(f);
    res_vec = zeros(maxit+1,1);
    res_vec(1) = rel_res;
    
    % initial iteration
    if rel_res > tol
        Ap = A*p;
        alpha = rTr_old/(p'*Ap);
        u = u + alpha*p;
        r = r - alpha*Ap;
        rTr_new = r'*r;
        rel_res = sqrt(rTr_new)/norm(f);
        res_vec(2) = rel_res;
        k = 1;
    else
        k = 0;
    end
    
    % more iterations
    while rel_res > tol && k < maxit
    
        % iteration counter
        k = k + 1;
    
        % updates
        beta = rTr_new/rTr_old;
        p = r + beta*p;
        rTr_old = rTr_new;
        Ap = A*p;
        alpha = rTr_old/(p'*Ap);
        u = u + alpha*p;
        r = r - alpha*Ap;
        rTr_new = r'*r;
        rel_res = sqrt(rTr_new)/norm(f);
        res_vec(k+1) = rel_res;
    
    end

    % cut tail of residual vector
    res_vec = res_vec(1:k+1);

end

% SSOR-PCG function
function [u, k, res_vec] = PCG_SSOR_function(A, D, M1, M2, f, u0, tol, maxit, omega)

    % initialization
    u = u0;
    r = f - A*u;
    rel_res = norm(r)/norm(f);
    res_vec = zeros(maxit+1,1);
    res_vec(1) = rel_res;
    z = omega*(2-omega) * (M2 \ (D*(M1 \ r)));
    p = z;
    rTz_old = r' * z;
    
    % initial iteration
    if rel_res > tol
        Ap = A*p;
        alpha = rTz_old/(p'*Ap);
        u = u + alpha*p;
        r = r - alpha*Ap;
        rel_res = norm(r)/norm(f);
        res_vec(2) = rel_res;
        k = 1;
    else
        k = 0;
    end
    
    % more iterations
    while rel_res > tol && k < maxit
    
        % iteration counter
        k = k + 1;
    
        % updates
        z = omega*(2-omega) * (M2 \ (D*(M1 \ r)));
        rTz_new = r'*z;
        beta = rTz_new/rTz_old;
        p = z + beta*p;
        rTz_old = rTz_new;
        Ap = A*p;
        alpha = rTz_old/(p'*Ap);
        u = u + alpha*p;
        r = r - alpha*Ap;
        rel_res = norm(r)/norm(f);
        res_vec(k+1) = rel_res;
        
    end
   
    % cut tail of residual vector
    res_vec = res_vec(1:k+1);

end