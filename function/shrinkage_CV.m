function [select_result,select_MSE,const] = shrinkage_CV(x,y,lambda,L_test,CV,const_flag)

for k = 1:CV
    L_test_each(k) = floor(L_test/CV);
end
L_test_each(k) = L_test - sum(L_test_each(1:CV-1));

num_lambda = length(lambda);
K = size(x,2);
%select_MSE = zeros(num_lambda,1);
select_MSE_all = zeros(num_lambda,CV);
select_result = zeros(K,num_lambda);
T = length(y);
if L_test >T
    error('Length of test sample is larger than the whole sample!')
end
for k = 1:CV
    train_size = [1:(T-sum(L_test_each(k:end)))];
    test_size = [(T-sum(L_test_each(k:end))+1):(T-sum(L_test_each(k:end))+L_test_each(k))];
    for n = 1:num_lambda    
        if const_flag ==0
            [select_result(:,n),info] = lasso(x,y,'Lambda',lambda(n),'Alpha',0.5,'Standardize',false,'Intercept',false);
            const(:,n) = 0;
        else
            [select_result(:,n),info] = lasso(x,y,'Lambda',lambda(n),'Alpha',0.5,'Standardize',false,'Intercept',true);
            const(:,n) = info.Intercept;
        end
%         select_temp = zeros(K,1);
%         select_temp(select_result(:,n)~=0) = 1./length(find(select_result(:,n)~=0));
        select_MSE_all(n,k) = mean((y(test_size) - x(test_size,:)*select_result(:,n) - const(:,n)).^2);
    end
end

select_MSE = mean(select_MSE_all,2);
% for n = 1:num_lambda
%     select_result(:,n) = lasso(x,y,'Lambda',lambda(n),'Alpha',0.5,'Intercept',false,'Standardize',false);
% end
end





