% clearing and closing
clc; close all; clear all; beep off;

% create scenarios
p_scenarios = [2,4,8,10,20];
l_scenarios = [1,2,4,10];
% l_scenarios = [0]; % for scenarios without overlap

% parameter settings
tol = 1e-10;
maxit = 10000;
Nruns_CPUtimes = 1; % set to 7 to estimate CPU times

% initialization
CPUtimes_AS_phase1 = zeros(length(p_scenarios),length(l_scenarios));
CPUtimes_AS_phase2 = zeros(length(p_scenarios),length(l_scenarios));
NumberIterations_AS = zeros(length(p_scenarios),length(l_scenarios));

% loop over scenarions
for pl = 1:length(p_scenarios)
    for ll = 1:length(l_scenarios)
    
        % parameter settings
        N = 200;
        h = 1/N;
        p_AS = p_scenarios(pl);
        l_AS = l_scenarios(ll);
        
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
        
        % set-up phase
        times = zeros(Nruns_CPUtimes,1);
        for rl = 1:Nruns_CPUtimes
        
            % start timer
            tic;

            % initialization
            idex_set = cell(p_AS,1);
            A_loc = cell(p_AS,1);
            CT_loc = cell(p_AS,1);
            R_loc = cell(p_AS,1);

            % loop over subdomains
            for q_AS = 1:p_AS
        
                % compute subdomain boundaries
                i_min = max(1, N/p_AS*(q_AS-1) + 1 - l_AS);
                i_max = min(N+1, N/p_AS*q_AS + l_AS);
        
                % compute indices
                idex_set{q_AS} = reshape((i_min:i_max).' + (0:N)*(N+1), [], 1);
        
                % construct sparse restriction matrix
                n_i = length(idex_set{q_AS});
                R_loc{q_AS} = sparse(1:n_i, idex_set{q_AS}, 1, n_i, (N+1)^2);
        
                % construct local matrix
                A_loc{q_AS} = R_loc{q_AS} * Ah * R_loc{q_AS}';
        
                % Sparse Cholesky factorization
                CT_loc{q_AS} = chol(A_loc{q_AS});
        
            end

            % store CPU times
            times(rl) = toc;
        
        end
        CPUtimes_AS_phase1(pl,ll) = median(times);
        
        % AS-PCG algorithm
        times = zeros(Nruns_CPUtimes,1);
        for rl = 1:Nruns_CPUtimes
            tic;
            [u_AS, k_AS, res_vec_AS] = PCG_AS_function(Ah, R_loc, CT_loc, p_AS, f_h(:), u0, tol, maxit);
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

% AS-PCG FUNCTION
function [u, k, res_vec] = PCG_AS_function(A, R_loc, CT_loc, p_AS, f, u0, tol, maxit)

    % initialization
    u = u0;
    r = f - A*u;
    rel_res = norm(r)/norm(f);
    res_vec = zeros(maxit+1,1);
    res_vec(1) = rel_res;
    z = Additive_Schwarz_function(r, R_loc, CT_loc,p_AS);
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
        z = Additive_Schwarz_function(r, R_loc, CT_loc,p_AS);
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

% Additive Schwarz function
function z = Additive_Schwarz_function(r, R_loc, CT_loc,p_AS)

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
