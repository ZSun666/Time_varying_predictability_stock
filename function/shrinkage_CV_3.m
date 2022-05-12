function [select_result,select_MSE] = shrinkage_CV_3(x,y,lambda,L_test,CV)

for k = 1:CV
    L_test_each(k) = floor(L_test/CV);
end
L_test_each(k) = L_test - sum(L_test_each(1:CV-1));

num_lambda = length(lambda);
MK = size(x,2);
%select_MSE = zeros(num_lambda,1);
select_MSE_all = zeros(num_lambda,CV);
select_result = zeros(MK,num_lambda);
T = length(y);
if L_test >T
    error('Length of test sample is larger than the whole sample!')
end
A = -eye(MK); b = zeros(MK,1);
Aeq = ones(1,MK); beq = 1; 
for k = 1:CV

    for n = 1:num_lambda    
        start_index = max(1,T-sum(L_test_each(k:end))-floor(lambda(n)));
        train_size = [start_index:(T-sum(L_test_each(k:end)))];      
        test_size = [(T-sum(L_test_each(k:end))+1):(T-sum(L_test_each(k:end))+L_test_each(k))];
        select_result(:,n) = lsqlin(x(train_size,:),y(train_size),A,b,Aeq,beq);
        select_result(select_result(:,n)<1e-4,n) = 0;
        select_temp = zeros(MK,1);
        select_temp(select_result(:,n)~=0) = 1./length(find(select_result(:,n)~=0));
        select_MSE_all(n,k) = mean((y(test_size)' - x(test_size,:)*select_temp).^2);
    end
end

select_MSE = mean(select_MSE_all,2);
% for n = 1:num_lambda
%     select_result(:,n) = lasso(x,y,'Lambda',lambda(n),'Alpha',0.5,'Intercept',false,'Standardize',false);
% end
end





