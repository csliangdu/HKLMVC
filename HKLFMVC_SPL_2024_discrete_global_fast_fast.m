function [Iabel, Ws, alpha, beta, objHistory] = HKLFMVC_SPL_2024_discrete_global_fast_fast(Hs, nCluster, LHs, Y)
nView = length(Hs);
nSmp = size(Hs{1}, 1);


%*************************************************
% Initialization of Ws
%*************************************************
Ws = cell(1, nView);
for iView = 1 : nView
    Ws{iView} = eye(size(Hs{iView}, 2), nCluster);
end

%*************************************************
% Initialization of alpha, beta
%*************************************************
beta = ones(nView, 1)/nView;
alpha = ones(nView, 1)/nView;
%*************************************************
% Initialization of Y
%*************************************************
if ~exist('Y', 'var')
    Ha = cell2mat(Hs);
    Ha = bsxfun(@rdivide, Ha, sqrt(sum(Ha.^2, 2)) + eps);
    label = litekmeans(Ha, nCluster, 'MaxIter', 50, 'Replicates', 10);
    Y = ind2vec(label')';
    clear Ha;
end

wLHs = cell(1, nView);
for iView = 1:nView
    wLH = zeros(nSmp, size(Hs{iView}, 2));
    for iView2 = 1:nView
        wLH = wLH + alpha(iView2) * LHs{iView, iView2};
    end
    wLHs{iView} = wLH;
end

iter = 0;
objHistory = [];
converges = false;
maxIter = 50;
while ~converges
    %***********************************************
    % Update Ws
    %***********************************************
    for iView = 1 : nView
        HLLH = wLHs{iView}' * wLHs{iView};
        HLLH = (HLLH + HLLH')/2;
        HLY = wLHs{iView}' * Y;
        Ws{iView} = updateWnew(HLLH, HLY, Ws{iView});
    end
    
    %***********************************************
    % Update Y
    %***********************************************
    wLHW = zeros(nSmp, nCluster);
    for iView = 1:nView
        wLHW = wLHW + (1/beta(iView)) * wLHs{iView} * Ws{iView}; % n d c m
    end
    [~, label] = max(wLHW, [], 2);
    Y = full(sparse(1:nSmp, label, ones(nSmp, 1), nSmp, nCluster));
    
    %***********************************************
    % Update alpha
    %***********************************************
    A = zeros(nView, nView);
    b = zeros(nView, 1);
    for iView = 1 : nView
        Ai = zeros(nView, nView);
        bi = zeros(nView, 1);
        LHWs = cell(1, nView);
        for iView2 = 1 : nView
            LHWs{iView2} = LHs{iView, iView2} * Ws{iView}; % n d c m^2
        end
        for iView2 = 1 : nView
            for iView3 = iView2 : nView
                Ai(iView2, iView3) = sum(sum( LHWs{iView2}.* LHWs{iView3})); % n c m^3
            end
            bi(iView2) = sum(sum( LHWs{iView2} .* Y));
        end
        Ai = max(Ai, Ai');
        A = A + (1/beta(iView)) * Ai;
        b = b + (1/beta(iView)) * bi;
    end
    opt = [];
    opt.Display = 'off';
    alpha_old = alpha;
    alpha = quadprog(A, -b, [], [], ones(1, nView), 1, zeros(nView, 1), ones(nView, 1), alpha_old, opt);
    
    wLHs = cell(1, nView);
    for iView = 1:nView
        wLH = zeros(nSmp, size(Hs{iView}, 2));
        for iView2 = 1:nView
            wLH = wLH + alpha(iView2) * LHs{iView, iView2};
        end
        wLHs{iView} = wLH;
    end
    
    %***********************************************
    % Update beta
    %***********************************************
    es = zeros(nView, 1);
    for iView = 1 : nView
        wLHW = wLHs{iView} * Ws{iView}; % n d c m
        E = wLHW - Y;
        es(iView) = sum(sum( E.^2 ));
    end
    beta = sqrt(es)/sum(sqrt(es));
    
    obj = sum(es./beta);
    objHistory = [objHistory; obj]; %#ok
    
    if iter > 2 && (abs((objHistory(iter-1)-objHistory(iter))/objHistory(iter-1))<1e-4)
        converges = 1;
    end
    
    if iter > maxIter
        converges = 1;
    end
    iter = iter + 1;
end
[~, Iabel] = max(Y, [], 2);
end

function [W, objHistory] = updateWnew(A, B, W)
%     min tr(W' A W) - 2 tr(W' B)
%     st W'W = I
n = size(A, 1);

if nargout > 1
    obj = trace(W' * A * W) - 2 * trace(W' * B);
    objHistory = obj;
end

iter = 0;
converges = false;
maxIter = 5;
tol = 1e-3;
max_iter = 10;
largest_eigenvalue = power_iteration(A, tol, max_iter);
% largest_eigenvalue = eigs(sparse(A), 1, 'largestreal');
Atau = largest_eigenvalue*eye(n) - A;
while ~converges
    W_old = W;
    AWB = 2 * Atau * W_old + 2 * B;
    [U,~,V] = svd(AWB, 'econ');
    W = U * V';
    val = norm(W - W_old,'inf');
    if nargout > 1
        obj = trace(W' * A * W) - 2 * trace(W' * B);
        objHistory = [objHistory; obj]; %#ok
    end
    
    if iter > 2 && abs(val) < 1e-3
        converges = 1;
    end
    if iter > maxIter
        converges = 1;
    end
    iter = iter + 1;
end

end

function largest_eigenvalue = power_iteration(A, tol, max_iter)
% A: Symmetric matrix
% tol: Tolerance for convergence
% max_iter: Maximum number of iterations

% Ensure the matrix is symmetric
if ~isequal(A, A')
    error('The matrix is not symmetric.');
end

% Initial guess for the eigenvector
n = size(A, 1);
b_k = rand(n, 1);

for k = 1:max_iter
    % Calculate the matrix-by-vector product Ab
    b_k1 = A * b_k;
    
    % Re-normalize the vector
    b_k1_norm = norm(b_k1);
    b_k = b_k1 / b_k1_norm;
    
    % Check for convergence
    if k > 1 && norm(b_k - b_k_prev) < tol
        break;
    end
    
    b_k_prev = b_k;
end

% Rayleigh quotient for the eigenvalue
largest_eigenvalue = (b_k' * A * b_k) / (b_k' * b_k);
end


function [obj, grad] = solveWnew(W, A, B)
%     min tr(W' A W) - 2 tr(W' B)
%     st W'W = I
%
AW = A * W;
grad = 2 * (AW - B);
obj1 = sum(sum(W .* AW));
obj2 = sum(sum(W .* B));
obj = obj1 - 2 * obj2;
end