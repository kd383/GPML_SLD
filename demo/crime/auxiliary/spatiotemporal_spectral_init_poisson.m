% Function to initialise SM kernel hyperparameters

% If varargin{1} is specified, it is the number of optimisation iterations
% to run for each random restart.  Otherwise, we will just have
% initialisations and no optimisation.

% stdy should be on the log-scale

function hyp = spatiotemporal_spectral_init(inf_method,hyp,meanf,lik,cov,covg,x,y,stdy,idx,nrestarts,varargin)
  
% start with user inputed hypers
  hyp_best = hyp;
  
  try
    % compute nlml for user specified hyp
    bestlik = Inf; %gp(hyp, inf_method, meanf, covg, lik, idx, y);
  catch
    disp('Error with user specified hypers.');
    disp('Attempting to proceed with automatic initialisation.');
    bestlik = Inf;
  end
  
  disp(sprintf('Initialisation nlml 0: %.02f',bestlik));
  if isnan(bestlik)
      bestlik = Inf;
  end
  % try 'nrestarts' number of initialisations
  for ri=1:nrestarts
    hyp.cov = [];     % shouldn't overwrite yet
    for i=1:numel(cov)
        % call the initialisation script for two 1D spectral mixture
        % kernels
                                                  % alert!: This assumes each
                                                  % separable component has
                                                  % the same Q!

        if strcmp(func2str(cov{i}{1}),'covSum')
            hyp.cov = [hyp.cov; hypinit1D(func2str(cov{i}{2}{2}{1}), cov{i}{2}{2}, x(:,i) ,stdy, covg{3}{i}(2) - covg{3}{i}(1))]; % want the true inputs x here
        else
            hyp.cov = [hyp.cov; hypinit1D(func2str(cov{i}{1}), cov{i}, x(:,i) ,stdy, covg{3}{i}(2) - covg{3}{i}(1))]; % want the true inputs x here
        end
    end
    hyp.cov = log(hyp.cov);
    
    % if desired, try iter_run optimization iterations for each initialisation
    if(~isempty(varargin))
          iter_run = varargin{1};
          hyp = minimize(hyp,@gp,-iter_run,inf_method,meanf,covg,lik,idx,y);
    end
    
    % see if nlml of new initialisation is better
    try
        l = gp(hyp, inf_method, meanf, covg, lik, idx, y);
        disp(sprintf('nlml %d: %.02f', ri, l))
        if l < bestlik
            bestlik = l;
            hyp_best = hyp;
            save('best.mat')
        end
    catch
        disp('Error trying initialisation');
    end
  end
  
  hyp = hyp_best;
  disp(sprintf('best likelihood found: %f', bestlik))

% initialise a 1D spectral mixture kernel
function [hypout] = hypinit1D(covtype,cov,x,stdy,Fs)
  hypout = [];
  switch(covtype)
    case 'covSM'
        Q = cov{2};
        % NOTE TO USER: SET FS= 1/[MIN GRID SPACING] FOR YOUR APPLICATION
        % Fs is the sampling rate
%        Fs = 1;   % 1/[grid spacing].  
             
        % Deterministic weights (fraction of variance)
        % Set so that k(0,0) is close to the empirical variance of the data.
        
        wm = stdy;
        w0 = 1/sqrt(Q)*ones(Q,1);
       
        w0 = w0.^2; % parametrization for covSMfast
        
        hypout = [w0];        
        
        % Uniform random frequencies
        % Fs/2 will typically be the Nyquist frequency
       % mu = max(Fs/2*rand(Q,1),1e-8);
        mu = abs(Fs/2/4*randn(Q,1));
        hypout = [hypout; mu];
        
        % Truncated Gaussian for length-scales (1/Sigma)
        maxlen = max(x) - min(x); % max. distance between any two points in input dimension
        % was maxlen = length(unique(x))
        sigmean = maxlen*sqrt(2*pi)/2; % 
        hypout = [hypout; 1./(abs(maxlen*randn(Q,1)))];
    case 'covMaterniso'
          %hypout = exp(randn(2,1));
          maxlen = max(x) - min(x); % max. distance between any two points in input dimension
          % was maxlen=length(unique(x)
          a = sqrt(maxlen)*sqrt(pi/2);
          hypout = [abs(a*randn);stdy];
    case 'covSEiso'
          %hypout = exp(randn(2,1));
          maxlen = max(x) - min(x); % max. distance between any two points in input dimension
          % was maxlen=length(unique(x)
          a = sqrt(naxlen)*sqrt(pi/2);
          hypout = [abs(a*randn);stdy];
    case 'covConstant'
          hypout = [];
  end
  % Todo: Add SE and MA kernels.
  

  
