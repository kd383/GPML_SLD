function demo_precip(method)
%
% Daily Precipitation Experiment
% Predict the US precipitation data from 2010
% Training points: 528474, Testing points: 100000
%
% method: Logdet Approximation Methods, {'lancz', 'SKI'}
%
% WARNING: This experiment takes quite long
%

if nargin < 1,  method = 'lancz'; end

% Recovery (a few hours)
% Use January data for hyper-parameter recovery
if 0 
    %Load and split data
    %===================
    
    [Xhyp, yhyp] = load_precip3240('data/precipitation3240/processed-data-2010-jan.csv');

    d = size(Xhyp, 2);
    limits = zeros(d, 2);
    limits(:,1) = min(Xhyp)';
    limits(:,2) = max(Xhyp)';

    % grid size for recovery
    ng = [100, 100, 50]; % lat, long, time

    % Setup MSGP
    % ==========
    xg = {};
    hyp = struct();
    hyp.cov = zeros(2*d, 1);
    cov  = {};
    sf = .5 * std(yhyp);
    for i=1:d
        spann = limits(i,2) - limits(i, 1);
        ell = spann / 20;
        hyp.cov(2 * (i-1)+1:2*i) = log([ell; sf^(1/d)]);
        cov{i} = {@covSEiso};
        xg{i} = {linspace(limits(i,1)-0.01*spann,limits(i, 2) + 0.01*spann,ng(i))'};
    end

    % set ldB2 options
    opt.cg_maxit = 1500;
    opt.cg_tol = 1e-4;
    opt.pred_var = 0;
    if strcmp(method, 'lancz')
        opt.ldB2_method = 'lancz';
        opt.ldB2_maxit = 50;
        opt.ldB2_hutch = sign(randn(size(Xhyp,1),5));
    end
    meanfunc = {@meanConst};
    lik = @likGauss; sn = .5 * std(yhyp);
    covg = {@covGrid,cov,xg};


    fprintf('----------Start Recovery----------\n');
    inf_method = @(varargin) infGrid(varargin{:},mopt);
    tic;
    hyp = minimize(hyp,@gp,-35, inf_method, meanfunc, covg, lik, Xhyp,yhyp);
    time = toc;
    fprintf('The recovered time is: %.4f [s]\n\n', time);
end

% Inference and Prediction (half an hour)
if 1 
    if strcmp(method,'lancz')
        a = load('precip_result.mat', 'hyp_lan');
        hyp = a.hyp_lan;
    else
        a = load('precip_result.mat', 'hyp_ski');
        hyp = a.hyp_ski;
    end
    

    ntest = 100000;
    [X, y, Xtest, ytest] = load_precip3240('data/precipitation3240/processed-data-2010.csv', ntest);
    ntrain = size(X, 1);

    d = size(X, 2);
    limits = zeros(d, 2);
    limits(:,1) = min(min(X), min(Xtest))';
    limits(:,2) = max(max(X), max(Xtest))';
    
    % Grid size for inference
    ng = [100, 100, 300]; % lat, long, time
    % Setup MSGP
    % ==========
    xg = {};
    cov  = {};
    sf = .5 * std(y);
    for i=1:d
        spann = limits(i,2) - limits(i, 1);
        cov{i} = {@covSEiso};
        xg{i} = {linspace(limits(i,1)-0.01*spann,limits(i, 2) + 0.01*spann,ng(i))'};
    end
    covg = {@covGrid,cov,xg};
    meanfunc = {@meanConst};
    lik = @likGauss;
    
    % set ldB2 options
    opt.cg_maxit = 800; 
    opt.cg_tol = 1e-4;
    opt.pred_var = 0;
    if strcmp(method,'lancz')
        opt.ldB2_method = 'lancz';
        opt.ldB2_maxit = -25;
        opt.ldB2_hutch = sign(randn(size(X,1),3));
    end

    fprintf('----------Start Inference----------\n');
    tic
    [post nlZ] = infGrid(hyp,meanfunc,covg,lik,X, y, opt);
    t_train = toc;
    fprintf('The inference time is: %.4f [s]\n\n', t_train);
    
    fprintf('----------Start Prediction----------\n');

    % Predict
    % =======
    tic;
    y_pred = post.predict(Xtest);
    t_predict = toc;
    mae = mean(abs(y_pred - ytest));
    mae_mp = mean(abs(bsxfun(@minus, ytest, mean(ytest))));
    smae = mae / mae_mp;
    mse = mean((y_pred - ytest).^2);
    fprintf('The SMAE is: %.4f.\n', smae);
    fprintf('The inference time is: %.4f [s]\n\n', t_predict);
end