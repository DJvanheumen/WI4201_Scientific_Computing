% clearing and closing
clc; close all; clear all; beep off;

% parameter settings
tol = 1e-10;
maxit = 10000;
omega = 1.5;
Nasympt = 5;
Nruns_CPUtimes = 1; % set to 7 to estimate CPU times

% create scenarios
N_scenarios = [40,80,120,160,200];
n_scenarios = (N_scenarios+1).^2;

% initialization
relres_hist = cell(1,length(N_scenarios));
alpha_mat = zeros(length(N_scenarios),Nasympt);
CPUtimes_SSOR = zeros(length(N_scenarios),1);
NumberIterations_SSOR = zeros(length(N_scenarios),1);

% loop over scenarios
for nl = 1: length(N_scenarios)

    % parameter settings
    N = N_scenarios(nl);
    h = 1/N;
    n = (N+1)^2;
    
    % initial guess
    u0 = zeros(n,1);
    
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
    fh = b_bdry;
    fh(2:N,2:N) = f_bar(2:N,2:N);
    fh = fh(:);
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
    D = spdiags(diag(Ah), 0, n, n);
    E = -tril(Ah,-1);
    F = -triu(Ah,1);
    
    % construct SSOR matrices
    M1 = D/omega - E;
    N1 = (1-omega)/omega*D + F;
    M2 = D/omega - F;
    N2 = (1-omega)/omega*D + E;
    
    % SSOR iterations
    times = zeros(Nruns_CPUtimes,1);
    for rl = 1:Nruns_CPUtimes
        tic;
        [uh, k, relres_vec] = SSOR_function(Ah, M1, M2, N1, N2, fh, u0, tol, maxit);
        times(rl) = toc;
    end
    CPUtimes_SSOR(nl) = median(times); 
    NumberIterations_SSOR(nl) = k;
    relres_hist{nl} = relres_vec;

end

% plot convergence history
figure;
semilogy(1:length(relres_hist{1}), relres_hist{1}, 'r-', 'LineWidth', 2);
hold on; semilogy(1:length(relres_hist{2}), relres_hist{2}, 'b-', 'LineWidth', 2);
hold on; semilogy(1:length(relres_hist{3}), relres_hist{3}, 'g-', 'LineWidth', 2);
hold on; semilogy(1:length(relres_hist{4}), relres_hist{4}, 'm-', 'LineWidth', 2);
hold on; semilogy(1:length(relres_hist{5}), relres_hist{5}, 'c-', 'LineWidth', 2);
xlabel('Iteration number', 'FontSize', 28);
ylabel('Relative residual', 'FontSize', 28);
set(gca, 'FontSize', 18, 'LineWidth', 1.2);
grid on; box on;
title('SSOR convergence history', 'FontSize', 28);
hold on;
yline(tol, 'k--', 'LineWidth', 2);
legend(sprintf('n = %d',n_scenarios(1)),sprintf('n = %d',n_scenarios(2)),sprintf('n = %d',n_scenarios(3)),sprintf('n = %d',n_scenarios(4)),sprintf('n = %d',n_scenarios(5)),'Tolerance','FontSize',20,'Location','best');

% compute residual ratios for asymptotic convergence
for il = 1:length(n_scenarios)
    for jl=1:Nasympt
    alpha_mat(il,jl) = relres_hist{il}(end-jl+1)/relres_hist{il}(end-jl);
    end
end

% plot number of iterations and CPU times
figure;
subplot(2,1,1);
xdat1 = n_scenarios(:);
ydat1 = NumberIterations_SSOR(:);
plot(0, 0, '-'); hold on;
plot(xdat1, ydat1, 'o-','LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k','Color','k');
xlabel('n', 'FontSize', 16);
ylabel('# SSOR iterations', 'FontSize', 16);
set(gca, 'FontSize', 16);
xlim([0 xdat1(end)*1.05]);
ylim([0 ydat1(end)*1.1]);
grid on;
subplot(2,1,2);
xdat2 = (n_scenarios(:)).^2;
ydat2 = CPUtimes_SSOR(:);
plot(0, 0, '-'); hold on;
plot(xdat2, ydat2, 'o-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k','Color','k');
xlabel('n^2', 'FontSize', 16);
ylabel('SSOR CPU time (s)', 'FontSize', 16);
set(gca, 'FontSize', 16);
xlim([0 xdat2(end)*1.05]);
ylim([0 ydat2(end)*1.1]);
grid on;

% print output
alpha_mat
NumberIterations_SSOR
CPUtimes_SSOR

% SSOR function
function [u, k, relres_vec] = SSOR_function(A, M1, M2, N1, N2, fh, u0, tol, maxit)

    % initialization
    k = 0;
    u = u0;
    relres = norm(fh-A*u,2)/norm(fh,2);
    relres_vec = [];
    
    % main loop
    while relres > tol && k < maxit
    
        % step 1: forward SOR sweep
        uh_half = M1 \ (N1*u + fh);
    
        % step 2: backward SOR sweep
        uh_new = M2 \ (N2*uh_half + fh);
    
        % update solution
        u = uh_new;
    
        % compute relative residual
        relres = norm(fh - A*u,2)/norm(fh,2);
        relres_vec = [relres_vec,relres];
    
        % update iteration counter
        k = k+1;
    
    end

end