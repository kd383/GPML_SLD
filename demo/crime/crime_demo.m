function [hyp0 hyp1] = crime_demo(method, hyp)
%
% Chicago Crime Modeling Experiment
% Predict the Chicago assault data 
% Training points: 126236, Testing points: 31408
%
% method: Logdet Approximation Methods, {'lancz', 'SKI'}
%

if nargin < 1,   method = 'lancz';   end

fprintf('---------Loading Data----------\n');
load('chicago_assault.mat');
if nargin >1, hyp0 = hyp; end
crime = crime(crime(:,1)>4.5,:); % exclude O'Hare
crime(:,3) = round(crime(:,3)/7);
timeperiod = max(crime(:,3));
% Use fisrt 417 weeks for training, last 104 weeks for testing
train_start_date = 0;
train_end_date = timeperiod - 52*2;
test_end_date = timeperiod;

% Grid discretization 
gridX = 17;
gridY = floor(1.55*gridX);

nstarts = 100; % sampling size for initial hyper-parameters
niters = 100; % optimizer iterations
Q = 20; % spectral mixture component

% rotate
theta = 10;
T = [cosd(theta) -sind(theta) 0; sind(theta) cosd(theta) 0; 0 0 1];
tform = affine2d(T);
[X,Y] = transformPointsForward(tform,crime(:,1),crime(:,2));
crime(:,1) = X;
crime(:,2) = Y;

train = crime(crime(:,3) <= train_end_date & crime(:,3) >= train_start_date,:);
test = crime(crime(:,3) > train_end_date & crime(:,3) <= test_end_date,:);
train_times = seqrange(train(:,3));
test_times = seqrange(test(:,3));

%count
Xleft = 8.15;  Xright = 22;  Ytop = 24;  Ybottom = -3.5;
xgrid = linspace(Xleft,Xright,gridX+1);
ygrid = linspace(Ybottom,Ytop,gridY+1);
[count,edges,mid,loc] = histcn(train,xgrid,ygrid,min(train_times):max(train_times));
[count_test,edges_test,mid_test,loc_test] = histcn(test,xgrid,ygrid,min(test_times):max(test_times));
cov = {{@covMaterniso,5},{@covMaterniso,5},{@covSM,Q}};
covg = {@covGrid,cov,{mid{1}',mid{2}',edges{3}'}};
xx = covGrid('expand', {mid{1}',mid{2}',edges{3}'});
yy = count(:);
expected = mean(count,3);
missing_Idx = expected(:)==0;
expected = log(expected(:));
expected(missing_Idx) = -100;
expected_train = repmat(expected,size(count,3),1);
expected_test = repmat(expected,length(train_times)+length(test_times),1);

xx_test = covGrid('expand',{mid_test{1}',mid_test{2}',edges_test{3}'});
yy_test = count_test(:);

xx = [xx repmat(expected,size(xx,1)/length(expected),1)];
xx_test = [xx_test repmat(expected,size(xx_test,1)/length(expected),1)];
xx_predict = [xx;xx_test];

Idx_train = find(xx(:,4)>-100);
Idx_test = find(xx_predict(:,4)>-100);
xx_predict = xx_predict(Idx_test,:);
ytrue = [yy; yy_test];
ytrue = ytrue(Idx_test);
yy = yy(Idx_train);
expected_train = expected_train(Idx_train);
expected_test = expected_test(Idx_test);

fprintf('Training points: %d, Testing points: %d\n\n',length(Idx_train), length(Idx_test)-length(Idx_train));

lik = {@likNegBinom,'exp'};
meanPop = @meanConst;
opt.cg_maxit = 500; opt.cg_tol = 1e-3;
if strcmp(method, 'lancz')
    opt.ldB2_method = 'lancz';
    opt.ldB2_hutch = sign(randn(size(xx,1),5));
    opt.ldB2_maxit = -30;
end

inf_method = @(varargin)infLaplace(varargin{:},opt);

% run this to find initial hyperparameter through a sampling process
% this can be very time consuming
% the output is loaded in by default
if 0
    hyps_init.mean = log(mean(yy)); 
    hyps_init.cov = zeros(4+3*Q,1);
    hyps_init.lik = -1;
    stdy = std(log(yy+1) - expected_train);
    tic;
    hyp0 = spatiotemporal_spectral_init_poisson(inf_method, hyps_init, meanPop, lik, cov, covg,xx,yy,stdy,Idx_train,nstarts);
    t = toc;
    fprintf('hyp0 complete in %.4f seconds.\n', t);
end

hyp0.lik = 1;
hyp0.cov([1 3]) = log([0.4 0.4]);
hyp0.mean = log(mean(yy));

fprintf('---------Start Recovery----------\n');
tic;
hyp1 = minimize_minfunc(hyp0, @gp, -niters, inf_method, meanPop, covg, lik, Idx_train, yy);
t = toc;
fprintf('Hyper-parameter recovered in %.4f seconds.\n\n',t);


covg{3} = {covg{3}{1},covg{3}{2},[train_times test_times]'};
fprintf('---------Start Prediction----------\n');
tic;
[post,nlZ,dnlZ] = infGrid(hyp1, {meanPop}, covg, lik, Idx_train,yy, opt);
[fmu,fs2,Z,vf] = post.predict(Idx_test);
t = toc;
fprintf('Prediction for both training and testing in %.6f seconds.\n',t);
n = length(Idx_train);
fprintf('Training error: %.4f\n',sqrt(mean((ytrue(1:n)-Z(1:n)).^2)));
fprintf('Testing error: %.4f\n',sqrt(mean((ytrue(n+1:end)-Z(n+1:end)).^2)));