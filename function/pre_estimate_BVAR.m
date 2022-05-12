function[beta_mean,beta_var,sigma] = pre_estimate_BVAR(Y,X)
    K = size(X,2);
    T = length(Y);
    % MN prior (but set all coefficients as 0)
    beta_prior = zeros(K,1);
    V_prior = eye(K);
    
    a_bar_1 = 0.001;
%     a_bar_2 = 0.001;
%     a_bar_3 = 1;
%     a_bar_4 = 0.001;
    sigma_sq = (var(Y));
   
    for j = 1:K   % for each j-th RHS variable
       
        V_prior(j,j) = a_bar_1; % variance on constant                
       
    end

    
    % posterior
    
    V_post = inv( inv(V_prior) + (1/sigma_sq)*X'*X ) ;
    beta_post = V_post * ( inv(V_prior)*beta_prior + (1/sigma_sq)*X'*Y );
    
    % In this case, the mean is a_post and the variance is V_post
    beta_mean=beta_post;
    beta_var = V_post;
    
    sigma = (Y - X*beta_post)'*(Y - X*beta_post)/(T-1);

end