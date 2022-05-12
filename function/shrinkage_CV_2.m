function [select_result,select_MSE] = shrinkage_CV_2(x,y,lambda,L_test)
num_lambda = length(lambda);
K = size(x,2);
select_MSE = zeros(num_lambda,1);
select_result = zeros(K,num_lambda);

A = -eye(K); b = zeros(K,1);
T = length(y);
if L_test >T
    error('Length of test sample is larger than the whole sample!')
end
for n = 1:num_lambda    
    select_result(:,n) = lsq_constrsparsereg(x(1:T-L_test,:),y(1:T-L_test),lambda(n), 'method', 'admm','A',A,'b',b);
     %= lasso(x(1:T-L_test,:),y(1:T-L_test),'Lambda',lambda(n),'Alpha',0.5,'Intercept',false,'Standardize',false);
    select_temp = zeros(K,1);
    select_temp(select_result(:,n)~=0) = 1./length(find(select_result(:,n)~=0));
    select_MSE(n) = mean((y(T-L_test+1:end)' - x(T-L_test+1:end,:)*select_temp).^2);
end


end





