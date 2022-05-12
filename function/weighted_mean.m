%% compute weithed mean
%% input Y vector ----- data
%% input W vector ----- weight

function[w_mean] = weighted_mean(Y,W)
    N = length(Y);
    w_mean = 0;
    for n = 1:N
        w_mean = w_mean+W(n)*Y(n);
    end
    
end