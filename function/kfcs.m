function [m_upd,P_upd,T_hat] = kfcs(y,Pi0,X,m_last,P_last,T_hat_last,Q,R,lambda)
m = size(X,2);

sig2init = Pi0;
%thresh_init = (3/4)*lambda * sigma0; %3*
thresh = 0.05;
%thresh = 3*(1/4)*lambda * sqrt(R); % 3*
thresh_del = thresh/3;
thresh_Pupd = 0.05*R;
k = 3; %duration to average x_upd to check for deletion

m_upd = zeros(m,1);
P_upd = zeros(m,m);


max_add = 2; %Smax/2 %1.25*Smax;

% Temporary KF 
% if isempty(T_hat_last)
%     Q1 = 0;
%     P_pred_tmp = 0;
%     K_tmp = 0;
%     mupd_tmp = 0;
%     P_upd_tmp = 0;
% %     m_upd = mupd_tmp;
% %     P_upd = P_upd_tmp;
%     y_fe =  y;
%     fe = y_fe'*y_fe;
% else
    Q1 = Q(T_hat_last,T_hat_last);
    P_pred_tmp = P_last(T_hat_last,T_hat_last) + Q1;
    K_tmp = P_pred_tmp*X(1,T_hat_last)'*inv(X(1,T_hat_last)*P_pred_tmp*X(1,T_hat_last)' + R);
    mupd_tmp = m_last(T_hat_last) + K_tmp*(y - X(1,T_hat_last)*m_last(T_hat_last));
    P_upd_tmp = (eye(length(T_hat_last)) - K_tmp*X(1,T_hat_last))*P_pred_tmp;

    m_upd(T_hat_last) = mupd_tmp;
    P_upd(T_hat_last,T_hat_last) = P_upd_tmp;
    % Compute filtering error (FE), FEN
    if ~isempty(T_hat_last)
        y_fe =  y - X(1,T_hat_last)*mupd_tmp;
    else
        y_fe = y;
    end
    Sigma_fe = (1 - X(1,T_hat_last)*K_tmp)*(X(1,T_hat_last)*P_pred_tmp*X(1,T_hat_last)' + R)*(1 - X(1,T_hat_last)*K_tmp)';
    fe = y_fe'*inv(Sigma_fe)*y_fe;
%end



T_hat = T_hat_last;
Tdiff_hat = [1];

while fe > 1.2 && ~isempty(Tdiff_hat)
    [~,beta_hat,~] = cs_solve(y_fe,X,1,sqrt(R),lambda,thresh); %change alpha
    T_hat_c = setdiff([1:m]',T_hat);
    Tdiff_hat = intersect(T_hat_c, find(abs(beta_hat) > thresh) );%Sdiff_hat = length(Tdiff_hat);
    %preventing too many additions: to ensure A(:,T_hat) remains full rank
    if length(Tdiff_hat) > max_add %1.25*
        Tdiff_hat0 = Tdiff_hat;
        disp('more than max_add directions detected'),
        [val,indx] = sort(abs(beta_hat));
        Tdiff_hat=indx(end:-1:end-max_add+1);
    end
    T_hat = [T_hat; Tdiff_hat]; 
    
    % Change Q1 after CS
    Q1 = zeros(m,m); 
    diffset = setdiff(T_hat,T_hat_last);
    commonset = intersect(T_hat,T_hat_last);
    Q1(diffset,diffset) = sig2init*eye(length(diffset));
    Q1(commonset,commonset) = Q(commonset,commonset);
    
    indx_new = sort([diffset;commonset]);
    
    % Final KF
    P_pred = P_last(indx_new,indx_new) + Q1(indx_new,indx_new);
    K = P_pred*X(1,indx_new)'*inv(R + X(1,indx_new)*P_pred*X(1,indx_new)');
    P_upd(indx_new,indx_new)  = ( eye(length(indx_new)) - K*X(1,indx_new) )*P_pred;
    m_pred = m_last(indx_new);
    m_upd(indx_new) = m_pred + K*(y -  X(1,indx_new)*m_pred);
    
    % Compute filtering error (FE), FEN
    y_fe =  y - X(1,indx_new)*m_upd(indx_new);
    Sigma_fe = (1 - X(1,indx_new)*K)*(X(1,indx_new)*P_pred*X(1,indx_new)' + R)*(1 - X(1,indx_new)*K)';
    fe = y_fe'*inv(Sigma_fe)*y_fe;

    %T_hat = T_hat_last;
    
end

    %% delete coefficient
    
    tmp = find(abs(m_upd)<thresh_del);
    
    Delta_c = intersect(T_hat, tmp );        
    T_hat = setdiff(T_hat, Delta_c);        
        
    Delta_r = Delta_c; %Delta_r = setdiff(T_hat_t{t-5},T_hat_t{t}) ;%intersect(T_hat_c,tmp2) % intersect(const_set, tmp2) 
    P_upd(Delta_r,[1:m]) = 0; 
    P_upd([1:m],Delta_r) = 0;
    m_upd(Delta_r) = 0;
    
end