function [bmean,bvar,amean,avar,ssigmean,sigvar] = ts_prior(Y,X,M,K,T)
    b_prior = 0*ones(M*K,1);   %<---- prior mean of alpha (parameter vector)
    Vb_prior = 1*eye(M*K);     %<---- prior variance of alpha
    
    % Hyperparameters on inv(SIGMA) ~ W(v_prior,inv(S_prior))
    v_prior = M;             %<---- prior Degrees of Freedom (DoF) of SIGMA
    S_prior = eye(M);            %<---- prior scale of SIGMA
    inv_S_prior = inv(S_prior);
    
    %-----------------------------PRELIMINARIES--------------------------------
    % First get ML estimators
    B_OLS = inv(X'*X)*(X'*Y); % This is the matrix of regression coefficients
    b_OLS = B_OLS(:);         % This is the vector of parameters, i.e. it holds
                              % that a_OLS = vec(A_OLS)
    SSE = (Y - X*B_OLS)'*(Y - X*B_OLS);   % Sum of squared errors
    SIGMA_OLS = SSE./(T-K+1);

    % Initialize Bayesian posterior parameters using OLS values
    beta = b_OLS;     % This is the single draw from the posterior of alpha
    Beta = Beta_OLS;     % This is the single draw from the posterior of ALPHA
    SSE_Gibbs = SSE;   % This is the single draw from the posterior of SSE
    SIGMA = SIGMA_OLS; % This is the single draw from the posterior of SIGMA
    
    for irep = 1:10000
    
    
    
    end
    
    
end