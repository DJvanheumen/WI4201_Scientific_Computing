% clearing and closing
clc; close all; clear all;

% set chapters
chapters = [true, true, true, true];

% global parameter setting
tol = 1e-10;
maxit = 10000;
omega = 1.5;
Nruns_CPUtimes = 1; % set to 1 for fast run; set to 7 to estimate CPU times more accurately

%%% START CHAPTER 1: DISCRETIZATION
if chapters(1) == true

    % set scenarios
    q_scenarios = 2:8;

    % initialization
    error_scenarios = zeros(length(q_scenarios),1);

    % loop over scenarios
    for il = 1:length(q_scenarios)

        % parameter settings
        N = 2^(q_scenarios(il));

        % discretization
        [h,n,u_ref,fh,Ah,~] = discretization_function(N);

        % direct solver by Cholesky factorization
        CT_Choleseky = chol(Ah);
        y_h = CT_Choleseky' \ fh(:);
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

end
%%% END CHAPTER 1: DISCRETIZATION

%%% START CHAPTER 2: SSOR
if chapters(2) == true

    % number of iterations for asymptotic analysis
    Nasympt = 5;

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
    
        % parameter setting
        N = N_scenarios(nl);

        % discretization
        [h,n,u_ref,fh,Ah,u0] = discretization_function(N);

        % Gauss-Seidel matrix splitting
        [~,~,~,M1,M2,N1,N2] = GaussSeidelSplitting_function(Ah,n,omega);
                
        % SSOR iterations
        times = zeros(Nruns_CPUtimes,1);
        for rl = 1:Nruns_CPUtimes
            tic;
            [uh,k,relres_vec] = SSOR_function(Ah,M1,M2,N1,N2,fh,u0,tol,maxit);
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

end
%%% END CHAPTER 2: SSOR

%%% START CHAPTER 3: CG AND SSOR-PCG
if chapters(3) == true
    
    % create scenarios
    N_scenarios = [40,80,120,160,200,240];
    n_scenarios = (N_scenarios+1).^2;
        
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

        % discretization
        [h,n,u_ref,fh,Ah,u0] = discretization_function(N);

        % Gauss-Seidel matrix splitting
        [D,~,~,M1,M2,N1,N2] = GaussSeidelSplitting_function(Ah,n,omega);
        
        % apply CG without preconditionig
        times = zeros(Nruns_CPUtimes,1);
        for rl = 1:Nruns_CPUtimes
        tic;
        [u_CG,k_CG,res_vec_CG] = CG_function(Ah,fh,u0,tol,maxit);
        times(rl) = toc;
        end
        CPUtimes_CG(nl) = median(times); 
        NumberIterations_CG(nl) = k_CG;
        u_CG = reshape(u_CG,N+1,N+1);
        
        % apply CG without with SSOR-conditionig
        times = zeros(Nruns_CPUtimes,1);
        for rl = 1:Nruns_CPUtimes
        tic;
        [u_PCG,k_PCG,res_vec_PCG] = PCG_SSOR_function(Ah,D,M1,M2,fh,u0,tol,maxit,omega);
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

end
%%% END CHAPTER 3: CG AND SSOR-PCG

%%% START CHAPTER 4: AS-PCG
if chapters(4) == true

    % parameter setting
    N = 200;

    % discretization
    [h,n,u_ref,fh,Ah,u0] = discretization_function(N);

    % create scenarios
    p_scenarios = [2,4,8,10,20];
    l_scenarios = [1,2,4,10];
    % l_scenarios = [0]; % for scenarios without overlap
    
    % initialization
    CPUtimes_AS_phase1 = zeros(length(p_scenarios),length(l_scenarios));
    CPUtimes_AS_phase2 = zeros(length(p_scenarios),length(l_scenarios));
    NumberIterations_AS = zeros(length(p_scenarios),length(l_scenarios));
    
    % loop over scenarions
    for pl = 1:length(p_scenarios)
        for ll = 1:length(l_scenarios)
        
            % parameter settings
            p_AS = p_scenarios(pl);
            l_AS = l_scenarios(ll);
            
            % set-up phase
            times = zeros(Nruns_CPUtimes,1);
            for rl = 1:Nruns_CPUtimes
                tic;
                [index_set,A_loc,CT_loc,R_loc] = setup_AS_function(Ah,N,p_AS,l_AS);
                times(rl) = toc;            
            end
            CPUtimes_AS_phase1(pl,ll) = median(times);
            
            % AS-PCG algorithm
            times = zeros(Nruns_CPUtimes,1);
            for rl = 1:Nruns_CPUtimes
                tic;
                [u_AS,k_AS,res_vec_AS] = PCG_AS_function(Ah,R_loc,CT_loc,p_AS,fh,u0,tol,maxit);
                times(rl) = toc;
            end
            CPUtimes_AS_phase2(pl,ll) = median(times); 
            NumberIterations_AS(pl,ll) = k_AS;
            
            % verify convergence to correct solution qualitatively
            %CT_Choleseky = chol(Ah);
            %y_h = CT_Choleseky' \ f_h(:);
            %u_h = CT_Choleseky \ y_h;
            %u_h = reshape(u_h, N+1, N+1);
            %clear y_h;
            %figure;subplot(2,2,1); imagesc(u_AS);
            %subplot(2,2,2); imagesc(u_h);
            
        end
    end
    
    % print output
    NumberIterations_AS
    CPUtimes_AS_phase1
    CPUtimes_AS_phase2

end
%%% END CHAPTER 4: AS-PCG

%%% FUNCTIONS

% start discretization function
function [h,n,u_ref,fh,Ah,u0] = discretization_function(N)

    % discretization
    h = 1 / N;
    n = (N+1)^2;
    
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
    
    % construct fh
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
    clear J_hat H_hat Th_hat Ih_hat X Y;

    % initial guess
    u0 = zeros(n,1);

end
% end discretization function

% start Gauss Seidel splitting function
function [D,E,F,M1,M2,N1,N2] = GaussSeidelSplitting_function(Ah,n,omega)

    % matrix splitting Ah = D - E - F
    D = spdiags(diag(Ah), 0, n, n);
    E = -tril(Ah,-1);
    F = -triu(Ah,1);
    
    % construct SSOR matrices
    M1 = D / omega - E;
    N1 = (1-omega) / omega*D + F;
    M2 = D / omega - F;
    N2 = (1-omega) / omega*D + E;

end
% end Gauss Seidel splitting function

% start SSOR function
function [u,k,relres_vec] = SSOR_function(Ah,M1,M2,N1,N2,fh,u0,tol,maxit)

    % initialization
    k = 0;
    u = u0;
    relres = norm(fh-Ah*u,2)/norm(fh,2);
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
        relres = norm(fh - Ah*u,2)/norm(fh,2);
        relres_vec = [relres_vec,relres];
    
        % update iteration counter
        k = k+1;
    
    end

end
% end SSOR function

% start CG function
function [u,k,res_vec] = CG_function(Ah,fh,u0,tol,maxit)

    % initialization
    u = u0;
    r = fh - Ah*u;
    p = r;
    rTr_old = r' * r;
    rel_res = norm(r)/norm(fh);
    res_vec = zeros(maxit+1,1);
    res_vec(1) = rel_res;
    
    % initial iteration
    if rel_res > tol
        Ap = Ah*p;
        alpha = rTr_old/(p'*Ap);
        u = u + alpha*p;
        r = r - alpha*Ap;
        rTr_new = r'*r;
        rel_res = sqrt(rTr_new)/norm(fh);
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
        Ap = Ah*p;
        alpha = rTr_old/(p'*Ap);
        u = u + alpha*p;
        r = r - alpha*Ap;
        rTr_new = r'*r;
        rel_res = sqrt(rTr_new)/norm(fh);
        res_vec(k+1) = rel_res;
    
    end

    % cut tail of residual vector
    res_vec = res_vec(1:k+1);

end
% end CG function

% start SSOR-PCG function
function [u,k,res_vec] = PCG_SSOR_function(Ah,D,M1,M2,fh,u0,tol,maxit,omega)

    % initialization
    u = u0;
    r = fh - Ah*u;
    rel_res = norm(r)/norm(fh);
    res_vec = zeros(maxit+1,1);
    res_vec(1) = rel_res;
    z = omega*(2-omega) * (M2 \ (D*(M1 \ r)));
    p = z;
    rTz_old = r' * z;
    
    % initial iteration
    if rel_res > tol
        Ap = Ah*p;
        alpha = rTz_old/(p'*Ap);
        u = u + alpha*p;
        r = r - alpha*Ap;
        rel_res = norm(r)/norm(fh);
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
        Ap = Ah*p;
        alpha = rTz_old/(p'*Ap);
        u = u + alpha*p;
        r = r - alpha*Ap;
        rel_res = norm(r)/norm(fh);
        res_vec(k+1) = rel_res;
        
    end
   
    % cut tail of residual vector
    res_vec = res_vec(1:k+1);

end
% end SSOR-PCG function

% start setup AS funcion
function [index_set,A_loc,CT_loc,R_loc] = setup_AS_function(Ah,N,p_AS,l_AS)

    % initialization
    index_set = cell(p_AS,1);
    A_loc = cell(p_AS,1);
    CT_loc = cell(p_AS,1);
    R_loc = cell(p_AS,1);
    
    % loop over subdomains
    for q_AS = 1:p_AS
    
        % compute subdomain boundaries
        i_min = max(1, N/p_AS*(q_AS-1) + 1 - l_AS);
        i_max = min(N+1, N/p_AS*q_AS + l_AS);
    
        % compute indices
        index_set{q_AS} = reshape((i_min:i_max).' + (0:N)*(N+1), [], 1);
    
        % construct sparse restriction matrix
        n_i = length(index_set{q_AS});
        R_loc{q_AS} = sparse(1:n_i, index_set{q_AS}, 1, n_i, (N+1)^2);
    
        % construct local matrix
        A_loc{q_AS} = R_loc{q_AS} * Ah * R_loc{q_AS}';
    
        % Sparse Cholesky factorization
        CT_loc{q_AS} = chol(A_loc{q_AS});
    
    end

end
% end setup AS function

% star AS-PCG FUNCTION
function [u,k,res_vec] = PCG_AS_function(Ah,R_loc,CT_loc,p_AS,fh,u0,tol,maxit)

    % initialization
    u = u0;
    r = fh - Ah*u;
    rel_res = norm(r)/norm(fh);
    res_vec = zeros(maxit+1,1);
    res_vec(1) = rel_res;
    z = Additive_Schwarz_function(r,R_loc,CT_loc,p_AS);
    p = z;
    rTz_old = r' * z;
    
    % initial iteration
    if rel_res > tol
        Ap = Ah*p;
        alpha = rTz_old/(p'*Ap);
        u = u + alpha*p;
        r = r - alpha*Ap;
        rel_res = norm(r)/norm(fh);
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
        z = Additive_Schwarz_function(r,R_loc,CT_loc,p_AS);
        rTz_new = r'*z;
        beta = rTz_new/rTz_old;
        p = z + beta*p;
        rTz_old = rTz_new;
        Ap = Ah*p;
        alpha = rTz_old/(p'*Ap);
        u = u + alpha*p;
        r = r - alpha*Ap;
        rel_res = norm(r)/norm(fh);
        res_vec(k+1) = rel_res;
    
    end
    
    % cut tail of residual vector
    res_vec = res_vec(1:k+1);

end

% Additive Schwarz function
function z = Additive_Schwarz_function(r,R_loc,CT_loc,p_AS)

    % initialization
    z = zeros(size(r));
    
    % loop over subdomains
    for q_AS = 1:p_AS
    
        % restriction
        r_i = R_loc{q_AS} * r;
    
        % solve locally
        z_i = CT_loc{q_AS} \ (CT_loc{q_AS}' \ r_i);
    
        % prolongation
        z = z + R_loc{q_AS}' * z_i;
    
    end

end
% end Additive Schwarz function
