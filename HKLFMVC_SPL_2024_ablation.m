function [Iabel, Ws, beta, objHistory] = HKLFMVC_SPL_2024_ablation(Hs, nCluster, Y)
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


iter = 0;
objHistory = [];
converges = false;
maxIter = 50;
while ~converges
    %***********************************************
    % Update Ws
    %***********************************************
    for iView = 1 : nView
        HH = Hs{iView}' * Hs{iView};
        HH = (HH + HH')/2;
        HY = Hs{iView}' * Y;
        Ws{iView} = updateWnew(HH, HY, Ws{iView});
    end
    
    %***********************************************
    % Update Y
    %***********************************************
    wHW = zeros(nSmp, nCluster);
    for iView = 1:nView
        wHW = wHW + (1/beta(iView)) * Hs{iView} * Ws{iView}; % n d c m
    end
    [~, label] = max(wHW, [], 2);
    Y = full(sparse(1:nSmp, label, ones(nSmp, 1), nSmp, nCluster));
    
    %***********************************************
    % Update beta
    %***********************************************
    es = zeros(nView, 1);
    for iView = 1 : nView
        wHW = Hs{iView} * Ws{iView}; % n d c m
        E = wHW - Y;
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
