function[X_out] = trans_norm(X_in)
K = size(X_in,2);

X_out = nan(size(X_in));
for k = 1: K
    available_index = find(~isnan(X_in(:,k)));
    
    if ~isempty(available_index)
        mean_temp = mean(X_in(available_index,k));
        sigma_temp = sqrt(var(X_in(available_index,k)));
        X_out(available_index,k) = (X_in(available_index,k)-mean_temp)./sigma_temp;
        
    end
end


end