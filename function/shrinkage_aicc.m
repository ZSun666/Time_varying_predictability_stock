function [select_result,select_aic,const] = shrinkage_aicc(x,y,lambda,const_flag)


num_lambda = length(lambda);
% num_X = size(x,2);

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
    log_lik = 0;
    sigma = info.MSE;
    for i = 1:size(y,1)
        log_lik = log_lik + log(mvnpdf(y(i),x(i,:)*select_result(:,n) + const(:,n),sigma));
    end
    num_X = max(1,length(find(select_result(:,n) ~=0)));
    [~,~,ic_temp] = aicbic(log_lik,num_X,i);
    aicc_temp = ic_temp.aicc;
    select_aic(n) = aicc_temp;
end


% select_MSE = mean(select_MSE_all,2);
% for n = 1:num_lambda
%     select_result(:,n) = lasso(x,y,'Lambda',lambda(n),'Alpha',0.5,'Intercept',false,'Standardize',false);
% end
end





