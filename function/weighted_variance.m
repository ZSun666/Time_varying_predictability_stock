%% compute weithed mean
%% input Y vector ----- data
%% input mean_Y   ----- weighted_mean of Y
%% input W vector ----- weight

function[w_var] = weighted_variance(Y,mean_Y,W)
    N = length(Y);
    w_var = 0;
    for n = 1:N
        w_var = w_var+W(n)*(Y(n)-mean_Y)^2;
    end
    
end