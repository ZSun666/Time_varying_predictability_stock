function [m,P] = ENKF_shrinkage_1(y,x,m_last,P_last,F,Q,sigma,N_en)

K = size(x,2);
m_draw = zeros(K,N_en);
m_update_draw = zeros(K,N_en);
y_draw = zeros(N_en,1);

for n = 1:N_en
    m_draw(:,n) = F * m_last + sqrt(P_last+Q)*randn(K,1);
    
end

m_predict = mean(m_draw,2);
P_predict = var(m_draw); % could add some kind of shrinkage at P_predict
K_gain = P_predict*x'/(x*P_predict*x'+sigma);

fix_update_temp = (eye(K)-K_gain*x);
for n = 1:N_en
    m_update_draw(:,n) = fix_update_temp * m_draw(:,n) + K_gain*(y+sqrt(sigma)*randn);
    
end
m = mean(m_update_draw,2);
P = var(m_update_draw,2);

end