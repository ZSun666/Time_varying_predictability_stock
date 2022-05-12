function[weight_restricted] = restricted_lasso_2(Y,X)
    %X = [ones(length(Y),1),X];
    MK = size(X,2);
    weight_restricted = zeros(MK,1);
    index_obs = find(~isnan(sum(X(end-60:end,:),1)));
    X = X(end-60:end,index_obs);
    Y = Y(end-60:end);
%     Y = Y(index_obs);
    H = 2*(X'*X); f = -2*Y'*X;
    
    MK = size(X,2);
    A(1:MK,:) = -eye(MK);
%     Aeq = ones(1,MK);
   % A(1,1) = 0;
    A(MK+1,:) = ones(1,MK);
%     Aeq = ones(1,MK);
    %Aeq(1) = 0;
    b = zeros(MK+1,1);
    b(1:MK) = zeros(MK,1);
    b(MK+1) = 1;
%     beq = 1;
%     beq = 1;
    
    opts = optimoptions('quadprog','Display','off','Algorithm','active-set');
    weight_restricted_temp = quadprog(H,f,A,b,[],[],[],[],zeros(MK,1),opts);
    weight_restricted_temp(abs(weight_restricted_temp) < 1e-3) = 0;
    weight_restricted(index_obs) = weight_restricted_temp(1:end);
    
    %A = -eye(MK); b = zeros(MK,1);
    
end