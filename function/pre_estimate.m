function[beta_mean_save,beta_var_save,sigma_save]...
        = pre_estimate(Y_pre,X_pre)
   
    
    K = size(X_pre,2);
    beta_mean_save = zeros(K,1);
    beta_var_save = 0.1*ones(K,1);
    for k = 1:K
        index_num = find(~isnan(X_pre(:,k)));
        X_m = X_pre(index_num,k);
        Y_m = Y_pre(index_num);
        
        beta_mean_save(k) = (X_m'*X_m)\(X_m'*Y_m);
        sigma_m(k) = (Y_m-X_m*beta_mean_save(k))'*(Y_m-X_m*beta_mean_save(k))/(length(Y_m)-1);
        beta_var_save(k) = sigma_m(k)/(X_m'*X_m);
        
    end
    sigma_save = mean(sigma_m(k));
end