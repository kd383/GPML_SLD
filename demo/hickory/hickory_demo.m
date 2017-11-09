function hickory_demo
%
% Hickory Tree Distribution Experiment
% Approximate the distribution of hickory tree in a square domain
% Number of data (tree): 703, Domain size: 60 x 60
%

set(0,'DefaultFigureColormap',jet);
load('hickory_data.mat');
figure('outerposition',[0 0 900 900])
subplot(2,2,1);
plot(xy(:,1),xy(:,2),'.')

% create the "computational grid"
n1 = 60; n2 = 60;
x1 = linspace(0,1.0001,n1+1); x2 = linspace(-0.0001,1,n2+1);
xg = {(x1(1:n1)+x1(2:n1+1))'/2, (x2(1:n2)+x2(2:n2+1))'/2};

i = ceil((xy(:,1)-x1(1))/(x1(2)-x1(1)));
j = ceil((xy(:,2)-x2(1))/(x2(2)-x2(1)));
counts = full(sparse(i,j,1));

% setup the GP
cv = {@covProd,{{@covMask,{1,@covSEiso}},{@covMask,{2,@covSEisoU}}}};
cvGrid = {@covGrid, { @covSEiso,  @covSEisoU}, xg};
hyp0.cov = log([.1  1 .1]);
y = counts(:);

X = covGrid('expand', cvGrid{3});
Idx = (1:length(y))';
lik = {@likPoisson, 'exp'};
hyp0.mean = .5;
hyp0.lik = [];
result = zeros(3,3);

% Exact
tic
% uncomment minimize for exact recovery, which may take a while
%hyp1 = minimize(hyp0, @gp, -60, @infLaplace, @meanConst, cv, lik, X, y);
hyp1 = hyp0; hyp1.cov = log([0.0629 0.6959 0.0851]); hyp1.mean = -1.8701;
result(1,3) = toc;

% Kronecker + Fiedler
opt1.cg_maxit = 600; opt1.cg_tol = 1e-4;
inf1 = @(varargin)infLaplace(varargin{:},opt1);
tic;
hyp2 = minimize(hyp0, @gp, -60, inf1, @meanConst, cvGrid, lik, Idx, y);
result(2,3) = toc;

% Kronecker + Lanczos
opt2.cg_maxit = 600; opt2.cg_tol = 1e-4; opt2.ldB2_method = 'lancz'; 
opt2.ldB2_hutch = sign(randn(3600,5)); opt2.ldB2_maxit = 25;
inf2 = @(varargin)infLaplace(varargin{:},opt2);
tic;
hyp3 = minimize(hyp0, @gp, -60, inf2, @meanConst, cvGrid, lik, Idx, y);
result(3,3) = toc;

% Actual log-likelihood
result(1,1) = gp(hyp1, @infLaplace, @meanConst, cv, lik, X, y);
result(2,1) = gp(hyp2, @infLaplace, @meanConst, cv, lik, X, y);
result(3,1) = gp(hyp3, @infLaplace, @meanConst, cv, lik, X, y);

% and what about the likelihood we actually calculated in infGrid_Laplace,
% using the Fiedler bound and Lanczos?
result(1,2) = result(1,1);
result(2,2) = gp(hyp2, inf1, @meanConst, cvGrid, lik, Idx, y);
result(3,2) = gp(hyp3, inf2, @meanConst, cvGrid, lik, Idx, y);

% let's compare our posterior predictions--to save time, we'll use infGrid_Laplace to
% make predictions in both cases (this won't hurt the accuracy, as we explain in the paper.)
[Ef1 Varf1 fmu1 fs1 ll1] = gp(hyp1, inf1, @meanConst, cvGrid, lik, Idx, y, Idx, y);
[Ef2 Varf2 fmu2 fs2 ll2] = gp(hyp2, inf1, @meanConst, cvGrid, lik, Idx, y, Idx, y);
[Ef3 Varf3 fmu3 fs3 ll3] = gp(hyp3, inf1, @meanConst, cvGrid, lik, Idx, y, Idx, y);

subplot(2,2,2);
imagesc([0 1], [0 1], (reshape(Ef1,60,60)'));
set(gca,'YDir','normal');
colorbar; 
title('Exact');
subplot(2,2,3);
imagesc([0 1], [0 1], (reshape(Ef2,60,60)'));
set(gca,'YDir','normal');
colorbar;
title('Fiedler');
subplot(2,2,4);
imagesc([0 1], [0 1], (reshape(Ef3,60,60)'));
set(gca,'YDir','normal');
colorbar;
title('Lanczos');

T = array2table(result,'VariableNames',{'True_LogLik','Apx_LogLik','Recovery_Time'},...
    'RowNames',{'Exact','Fiedler','Lanczos'});
display(T)