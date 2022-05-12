function[X_out] = trans_norm_2(X_in,X_mean,X_var)
K = size(X_in,2);
for k = 1: K
    
    X_out(:,k) = (X_in(:,k)-X_mean(k))/sqrt(X_var(k));
end


end