function [X,y, Xtest, ytest] = load_precip3240(filename, ntest, ntrain, seed)
if nargin < 4
    seed = 23;
end
s = RandStream('mt19937ar','Seed',seed);
data = csvread(filename, 1, 4);
m = size(data, 1);
p = randperm(s, m);

if nargin < 2
    y = data(:,end);
    X = data(:, 1:end-1);
else
    if nargin <= 2
        ntrain = m - ntest;
    end
    y = data(p(ntest+1:ntest+ntrain),end);
    X = data(p(ntest+1:ntest+ntrain), 1:end-1);

    ytest = data(p(1:ntest),end);
    Xtest = data(p(1:ntest), 1:end-1);
end
