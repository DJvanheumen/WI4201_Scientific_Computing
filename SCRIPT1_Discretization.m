% clearing and closing
clc; close all; clear all; beep off;

% scenarios
q_scenarios = 2:8;

% initialization
error_scenarios = zeros(length(q_scenarios),1);

% loop over scenarios
for il = 1:length(q_scenarios)

    % parameter settings
    N = 2^(q_scenarios(il));
    h = 1 / N;

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

    % direct solver by Cholesky factorization
    CT_Choleseky = chol(Ah);
    y_h = CT_Choleseky' \ f_h(:);
    u_h = CT_Choleseky \ y_h;
    u_h = reshape(u_h, N+1, N+1);
    clear y_h;

    % store errors
    error_scenarios(il) = max(abs(u_h(:) - u_ref(:)));

end

% plot convergence
xvals = 2.^(-q_scenarios);
yvals = error_scenarios.';
figure; loglog(xvals, yvals, 'o-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k','Color','k');
hold on;
set(gca, 'FontSize', 24);
xlabel('$h$', 'Interpreter', 'latex', 'FontSize', 32);
ylabel('$\|{\bf{u}}^h-{\bf{u}}^h_{\mathrm{ex}}\|_\infty$','Interpreter', 'latex', 'FontSize', 24);
title('Discretization error', 'FontSize', 32);
grid on; box on; 

% estimate order of convergence
beta_estimates = log(yvals(end)./yvals(1:end-1)) ./ log(xvals(end)./xvals(1:end-1));

% print output
beta_estimates