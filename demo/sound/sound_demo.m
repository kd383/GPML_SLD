function sound_demo(method, ninterp, hyp)
%
% Natural Sound Recovery Experiment
% Recover contiguous missing region in a waveform using different 
% Training points: 59306, Testing points: 691
%
% method: Logdet Approximation Methods, {'Lanczos', 'Cheby', 'SKI', 'FITC'}
% ninterp: Number of interpolation points, default: 3000
% hyp: Initial hyper-parameters.
%
if nargin < 1 || isempty(method), method = 'Lanczos'; end       % Default method: Lanczos
if nargin < 2 || isempty(ninterp),    ninterp = 3000; end       % Default Gridpts: 3000
if nargin < 3,  hyp = [];   end

% Loading Sound Data
fprintf('---------Loading Data----------\n');
testVars = {'xfull','xtrain','ytrain','xtest','ytest','hypc'};
load('audio_data.mat',testVars{:});

Ntrain = length(xtrain);
Nrun = numel(ninterp);

% Initialize Error/Time 
MAE = zeros(Nrun,1);
SMAE = zeros(Nrun,1);
RMSE = zeros(Nrun,1);
TRIVMAE = zeros(Nrun,1);
TRECOVER = zeros(Nrun,1);
TINFER = zeros(Nrun,1);
TPRED = zeros(Nrun,1);

meanfunc = {@meanZero};

fprintf('Using Method: %s\n\n', method);
for i = 1:Nrun
    xg = covGrid('create',xfull,1,ninterp(i));              % Create grid points
    [inf, covg, opt] = build_inf(method, xg, Ntrain);       % Load options based on method
    if strcmp(method, 'FITC'),  hyp = hypc;  end            % No recovery using FITC
    if isempty(hyp)
        fprintf('----------Start Recovery----------\n');
        tic;
        hyp = minimize(hypc,@gp,-50,inf,meanfunc,covg,'likGauss',...
                       xtrain,ytrain);                      % Recover hyper-params
        fprintf('The recovered hyper-parameters are:\n');
        fprintf('ell = %.4f,\t sf = %.4f,\t sigma = %.4f\n',...
                 [exp(hyp.cov)',exp(hyp.lik)]);
        TRECOVER(i) = toc;
        fprintf('The recovered time is: %.4f [s]\n\n', TRECOVER(i));
    end
    
    fprintf('----------Start Inference----------\n');
    tic;
    if strcmp(method, 'FITC')
        [post, nlZ, dnlZ] = infFITC(hyp,meanfunc,covg,'likGauss',...
                                xtrain,ytrain,opt);
    else
        [post, nlZ, dnlZ] = infGrid(hyp,meanfunc,covg,'likGauss',...
                                xtrain,ytrain,opt);
    end
    TINFER(i) = toc;
    fprintf('The inference time is: %.4f [s]\n\n', TINFER(i));
            
    fprintf('----------Start Prediction----------\n');
    post.L = @(x) 0*x;                                         % Fast prediction
    tic;
    ymug = gp(hyp,inf,meanfunc,covg,[],xtrain,post,xtest);
    TPRED(i) = toc;
            
    MAE(i)  = sum(abs(ytest-ymug))/numel(ytest);
    RMSE(i) = sqrt(sum((ytest-ymug).^2)/numel(ytest));
    TRIVMAE(i) = sum(abs(ytest-zeros(numel(ytest),1)))/numel(ytest);
    SMAE(i) = MAE(i)./TRIVMAE(i);
    fprintf(['MAE = %5.3e, TRIVMAE = %5.3e, SMAE = %5.3e,\n', ...
             'RMSE = %5.3e, time = %5.3f [s]\n\n'], MAE(i), TRIVMAE(i), ...
              SMAE(i), RMSE(i), TPRED(i));
end

end

% Loading options into inference and covariance function
function [inf, covg, opt] = build_inf(method, xg, Ntrain)

opt.cg_maxit = 1e4; opt.cg_tol = 1e-2;                       % CG solver options
switch method
    case 'Lanczos'
        opt.ldB2_method = 'lancz'; 
        opt.ldB2_maxit = 25;                                     % Lanczos steps
        opt.ldB2_hutch = sign(randn(Ntrain,5));        % Number of probe vectors
        inf = @(varargin) infGrid(varargin{:},opt);
        covg = {@covGrid,{@covSEiso},xg};
    case 'Cheby'
        opt.ldB2_method = 'cheby';
        opt.ldB2_hutch = 5;                            % Number of probe vectors
        opt.ldB2_maxit = 1e3;                    % Number of iterations for eigs
        opt.ldB2_cheby_degree = 100;                % Degree of Cheby polynomial
        inf = @(varargin) infGrid(varargin{:},opt);
        covg = {@covGrid,{@covSEiso},xg};
    case 'SKI'
        inf = @(varargin) infGrid(varargin{:},opt);
        covg = {@covGrid,{@covSEiso},xg};
    case 'FITC'
        inf = @infFITC;
        covg = {@covFITC,{@covSEiso},xg{1}};
end

end