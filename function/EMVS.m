function [select_index] = EMVS(x,y,sigma,select_init)
    [T,K] = size(x);
    theta_n = 0.05;
    gamma_n = zeros(K,1);
    gamma_n(select_init ~= 0) = 1;
    gamma_last = gamma_n;
    v_0 = 0.001;% spike
    v_1 = 100; % slab
    a_0 = 2;  b_0 = 10;
    lambda = 1; nu = 1;
    d_n = gamma_n*v_1+(1-gamma_n)*v_0;
    V_n = (x'*x+diag(1./d_n));
    inv_V_n = inv(V_n);
    xy = x'*y;
    m_n = inv_V_n*xy;
    sigma = mean((y-x*m_n).^2);
    
    count_k_0 = 0;
    count_iter = 1;
    while count_k_0 <3 && count_iter < 100
       
        
        
        r_temp = (sigma/(1/v_0 - 1/v_1))*(log(v_1/v_0)-2*log(theta_n/(1-theta_n)));
        for k = 1:K
            E_beta_2(k,1) = m_n(k)^2+ sigma*inv_V_n(k,k);
            if  E_beta_2(k) > r_temp
                gamma_n(k) = 1;
            else
                gamma_n(k) = 0;
            end
        end
        theta_n = (sum(gamma_n)+a_0-1)/(K+a_0+b_0-2);
        index_diff = find(gamma_n~=gamma_last);
        if ~isempty(index_diff)
            inv_d_diff = zeros(K,1);
            d_n(index_diff) = d_n(index_diff) + (v_1-v_0)*(2*gamma_n(index_diff)-1);
            inv_d_diff(index_diff) = (1/v_1 - 1/v_0)*(2*gamma_n(index_diff)-1); %%---------------------
            V_n = V_n + diag(inv_d_diff);
            inv_V_n = inv(V_n);
            m_n = inv_V_n * xy;
            for k = 1:K
                E_beta_2(k,1) = m_n(k)^2+ sigma*inv_V_n(k,k);
            end
             E_sse = sigma *trace(x*inv_V_n*x')+sum((y-x*m_n).^2);
             E_beta_2_d = E_beta_2./d_n;
             sigma = (E_sse+sum(E_beta_2_d)+ nu *lambda)/(T+K+nu);
        else
            count_k_0 = count_k_0+1;
        end
        
        
        gamma_last = gamma_n;
        count_iter = count_iter +1;
    end
    select_index = gamma_n;
end