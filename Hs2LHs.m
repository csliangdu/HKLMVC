function [LHs, LHLHs] = Hs2LHs(Hs, eta, knn_size)
nView = length(Hs);
nSmp = size(Hs{1}, 1);

% Ss = cell(1, nView);
% Ls = cell(1, nView);
% HLs = cell(1, nView);
LHs = cell(nView, nView);
for iView = 1:nView
    S = constructW_PKN_du(Hs{iView}', knn_size, 1);
    % Ss{iView} = S;
    DS = sum(S, 2).^(-.5);
    S = bsxfun(@times, S, DS);
    S = bsxfun(@times, S, DS');
    L = speye(nSmp) - S;
    L = (L + L')/2;
    % Ls{iView} = L;
    HL = expm(-eta * L);
    % HLs{iView} = HL;
    for iView2 = 1:nView
        LHs{iView2, iView} = HL * Hs{iView2};
    end
end
clear S DS L HL;
if nargout > 1
    nDim = size(LHs{1,1}, 2);
    LHLHs = cell(nView, 1);
    for iView1 = 1:nView
        
        idx = 0;
        ABs = zeros((nDim*nDim), nView*(nView+1)/2);
        for iView2 = 1:nView
            LHa = LHs{iView1, iView2};
            for iView3 = iView2:nView
                LHb = LHs{iView1, iView3};
                idx = idx + 1;
                LHaLHb = LHa' * LHb;
                ABs(:, idx) = LHaLHb(:);
            end
        end
        LHLHs{iView1} = ABs;
    end
end

end