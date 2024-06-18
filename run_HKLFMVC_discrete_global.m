%
%
%

clear;
clc;
data_path = fullfile(pwd, '..',  filesep, "data_Hs", filesep);
addpath(data_path);
lib_path = fullfile(pwd, '..',  filesep, "lib", filesep);
addpath(lib_path);
code_path = genpath(fullfile(pwd, '..',  filesep, 'HKLFMVC-SPL-2024'));
addpath(code_path);

dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};

data_path_Hs = fullfile(pwd, '..',  filesep, "data_Hs", filesep);

exp_n = 'HKLFMVC_discrete_global';
% profile off;
% profile on;
for i1 = 1:length(datasetCandi)
    data_name = datasetCandi{i1}(1:end-4);
    dir_name = [pwd, filesep, exp_n, filesep, data_name, filesep];
    create_dir(dir_name);
    fname2 = fullfile(data_path_Hs, [data_name, '.mat']);
    load(fname2);
    nCluster = length(unique(Y));
    nView = length(Hs);
    nDim = nCluster * 4;
    
    
    embeddings_s = [1, 2, 4]; % default
    eta_s = [9]; % default
    knn_size_s = [10]; % default
    paramCell = cell(1, length(embeddings_s) * length(eta_s) * length(knn_size_s));
    idx = 0;
    for iParam1 = 1:length(embeddings_s)
        for iParam2 = 1:length(eta_s)
            for iParam3 = 1:length(knn_size_s)
                idx = idx + 1;
                param = [];
                param.nEmbedding = embeddings_s(iParam1);
                param.eta = eta_s(iParam2);
                param.knn_size = knn_size_s(iParam3);
                paramCell{idx} = param;
            end
        end
    end
    paramCell = paramCell(~cellfun(@isempty, paramCell));
    nParam = length(paramCell);
    
    nMeasure = 13;
    nRepeat = 10;
    seed = 2024;
    rng(seed);
    % Generate 50 random seeds
    random_seeds = randi([0, 1000000], 1, nRepeat);
    % Store the original state of the random number generator
    original_rng_state = rng;
    %*********************************************************************
    % HKLFMVC_discrete_global
    %*********************************************************************
    fname2 = fullfile(dir_name, [data_name, '_HKLFMVC_discrete_global.mat']);
    if ~exist(fname2, 'file')
        HKLFMVC_discrete_global_result = zeros(nParam, 1, nRepeat, nMeasure);
        HKLFMVC_discrete_global_time = zeros(nParam, 1);
        for iParam = 1:nParam
            param = paramCell{iParam};
            nEmbedding = param.nEmbedding * nCluster;
            eta = param.eta;
            knn_size = param.knn_size;
            Hs_new = cell(1, nView);
            for iKernel = 1:nView
                Hi = Hs{iKernel};
                Hs_new{iKernel} = Hi(:, 1: nEmbedding);
            end
            t1_s = tic;
            if eta > 0
                LHs = Hs2LHs(Hs_new, eta, knn_size);
            else
                LHs = Hs;
            end
            for iRepeat = 1:nRepeat
                % Restore the original state of the random number generator
                rng(original_rng_state);
                % Set the seed for the current iteration
                rng(random_seeds(iRepeat));
                Ha = cell2mat(Hs_new);
                Ha = bsxfun(@rdivide, Ha, sqrt(sum(Ha.^2, 2)) + eps);
                label_0 = litekmeans(Ha, nCluster, 'MaxIter', 50, 'Replicates', 10);
                Y_0 = ind2vec(label_0')';
                [Iabel, Ws, alpha, beta, objHistory] = HKLFMVC_SPL_2024_discrete_global_fast_fast(Hs_new, nCluster, LHs, Y_0);
                result_aio = my_eval_y(Iabel, Y);
                HKLFMVC_discrete_global_result(iParam, 1, iRepeat, :) = result_aio';
            end
            t1 = toc(t1_s);
            HKLFMVC_discrete_global_time(iParam) = t1/nRepeat;
        end
        a1 = sum(HKLFMVC_discrete_global_result, 2);
        a3 = sum(a1, 3);
        a4 = reshape(a3, nParam, nMeasure);
        a4 = a4/nRepeat;
        HKLFMVC_discrete_global_result_summary = [max(a4, [], 1), sum(HKLFMVC_discrete_global_time)];
        save(fname2, 'HKLFMVC_discrete_global_result', 'HKLFMVC_discrete_global_time', 'HKLFMVC_discrete_global_result_summary');
        
        disp([data_name, ' has been completed!']);
    end
end
% profile viewer;
rmpath(data_path);
rmpath(lib_path);
rmpath(code_path);