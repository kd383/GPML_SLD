%clearvars -except 
%close all
set(0,'DefaultFigureColormap',jet);

% load the hickory data
load('hickory_data.mat');
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
% lik = @likGauss;
hyp0.mean = .5;
% hyp0.lik = -2.3;
hyp0.lik = [];

% Exact
tic
%hyp1 = minimize(hyp0, @gp, -100, @infLaplace, @meanConst, cv, lik, X, y);
time1 = toc
exp(hyp1.cov)

% Kronecker + Fiedler
opt1.cg_maxit = 600; opt1.cg_tol = 1e-4;
inf1 = @(varargin)infLaplace(varargin{:},opt1);
tic
hyp2 = minimize(hyp0, @gp, -100, inf1, @meanConst, cvGrid, lik, Idx, y);
time2 = toc
exp(hyp2.cov)

% Kronecker + Lanczos
opt2.cg_maxit = 600; opt2.cg_tol = 1e-4; opt2.ldB2_method = 'lancz'; 
opt2.ldB2_hutch = sign(randn(3600,4)); opt2.ldB2_maxit = 20;
inf2 = @(varargin)infLaplace(varargin{:},opt2);
tic
hyp3 = minimize(hyp0, @gp, -100, inf2, @meanConst, cvGrid, lik, Idx, y);
time3 = toc
exp(hyp3.cov)

% Actual log-likelihood
fprintf('Log-likelihood learned w/out Kronecker: %.04f\n', gp(hyp1, @infLaplace, @meanConst, cv, lik, X, y));
fprintf('Log-likelihood learned with Kronecker + Fiedler: %.04f\n', gp(hyp2, @infLaplace, @meanConst, cv, lik, X, y));
fprintf('Log-likelihood learned with Kronecker + Lanczos: %.04f\n', gp(hyp3, @infLaplace, @meanConst, cv, lik, X, y));

% and what about the likelihood we actually calculated in infGrid_Laplace,
% using the Fiedler bound?
fprintf('Log-likelihood with Kronecker + Fiedler bound: %.04f\n', gp(hyp2, inf1, @meanConst, cvGrid, lik, Idx, y));
fprintf('Log-likelihood with Kronecker + Lanczos: %.04f\n', gp(hyp3, inf2, @meanConst, cvGrid, lik, Idx, y));

% let's compare our posterior predictions--to save time, we'll use infGrid_Laplace to
% make predictions in both cases (this won't hurt the accuracy, as we explain in the paper.)
[Ef1 Varf1 fmu1 fs1 ll1] = gp(hyp1, inf1, @meanConst, cvGrid, lik, Idx, y, Idx, y);
[Ef2 Varf2 fmu2 fs2 ll2] = gp(hyp2, inf1, @meanConst, cvGrid, lik, Idx, y, Idx, y);
[Ef3 Varf3 fmu3 fs3 ll3] = gp(hyp3, inf1, @meanConst, cvGrid, lik, Idx, y, Idx, y);

subplot(1,3,1); imagesc([0 1], [0 1], (reshape(Ef1,60,60)')); set(gca,'YDir','normal'); colorbar; 
title('Exact'); xlabel(sprintf('Runtime: %.4f',time1));
subplot(1,3,2); imagesc([0 1], [0 1], (reshape(Ef2,60,60)')); set(gca,'YDir','normal'); colorbar;
title('Fiedler'); xlabel(sprintf('Runtime: %.4f',time2));
subplot(1,3,3); imagesc([0 1], [0 1], (reshape(Ef3,60,60)')); set(gca,'YDir','normal'); colorbar;
title('Lanczos'); xlabel(sprintf('Runtime: %.4f',time3));