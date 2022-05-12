function[beta_mean_save,beta_var_save,const_save,sigma_save]...
        = pre_estimate_3(Y_pre,X_pre)
   
    
    K = size(X_pre,2);
    beta_mean_save = zeros(K,1);
    beta_var_save = 0.001*ones(K,1);
    
        index_num = find(~isnan(sum(X_pre,2)));
        X_m = X_pre(index_num,:);
        Y_m = Y_pre(index_num);
        
%         beta_mean_save = lasso(X_m,Y_m,'Lambda',1,'Alpha',0.5,'Intercept',false,'Standardize',false);
        [select_result,select_MSE] = shrinkage_CV_1(X_m,Y_m,[1:0.1:2],ceil(length(Y_m)/2),5);
        [~,min_index] = min(select_MSE);
        beta_mean_save = mean(select_result(2:end,min_index),2);
        const_save = mean(select_result(1,min_index),2);
       % lasso(x(train_size,:),y(train_size),'Lambda',1,'Alpha',0.5,'Intercept',false,'Standardize',false);
        sigma_save = (Y_m-X_m*beta_mean_save-const_save)'*(Y_m-X_m*beta_mean_save-const_save)/(length(Y_m)-1);
        %beta_var_save(k) = sigma_m(k)/(X_m'*X_m);
       
end