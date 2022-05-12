function[m_restricted,p_restricted] = restricted_KF_l1(Y,X,m_last,P_last,Q,R)
    MK = size(m_last,1);
%     Y_augment = [Y;1];
%     X_augment = [X;ones(1,MK)];
%     R_augment = [R;0];
    
    m_predict = m_last;
    P_predict = P_last + Q;
    P_predict = 0.5*(P_predict'+P_predict);
    % update step
%     f_temp = X_augment * P_predict * X_augment'+ R_augment;
%     KG = P_predict *X_augment' / f_temp; % kalman gain
%     m_update = m_predict + KG*(Y_augment-X_augment * m_predict);
%     P_update = (eye(MK) - KG*X_augment)* P_predict;
    f_temp = X * P_predict * X'+ R;
    KG = P_predict *X' / f_temp; % kalman gain
    m_update = m_predict + KG*(Y-X * m_predict);
    P_update = (eye(MK) - KG*X)* P_predict; 
%     

    A = zeros(MK+1,MK);  b = zeros(MK+1,1);
    A = -eye(MK); b = zeros(MK,1); A(MK+1,:) = ones(1,MK);b(MK+1,:) = 0.1;
  % Aeq = ones(1,MK); beq = 1; 
% %     A = zeros(MK*3,MK*2);
% %     A(1:MK,1:end) = [eye(MK),eye(MK)];
% %     A(MK+1:end,1:MK) = [-eye(MK);-eye(MK)];
% %     b = zeros(3*MK,1);
%     
    H = eye(MK)*2; f = -2*m_update';% + 0.5*ones(1,MK) ;
    opts = optimoptions('quadprog','Display','off','Algorithm','active-set');
    m_restricted = quadprog(H,f,A,b,[],[],[],[],m_update,opts);
    m_restricted(m_restricted < 1e-3) = 0; 
    p_restricted = P_update;
%     H = 2*eye(MK)*(X*P_predict*X'+ R);f=-2*X*P_predict;
%    opts = optimoptions('quadprog','Display','off','Algorithm','active-set');
%     K_restricted = quadprog(H,f,A,b,Aeq,beq,[],[],KG,opts);
%     %K_restricted(K_restricted<5e-3) = 0;
%     m_restricted = m_predict + K_restricted*(Y-X * m_predict);
%     p_restricted = (eye(MK) - K_restricted*X)* P_predict; 
%     p_restricted = (p_restricted'+p_restricted)/2;
%     m_restricted(m_restricted < 5e-3) = 0; 
    %m_restricted(m_restricted ~= 0) = 1/length(find(m_restricted~=0));
%     ob_function = @(x)(((x-m_update)'))*(x-m_update);
%     opts = optimoptions('fmincon','Display','off','Algorithm','sqp');
%     m_restriced = fmincon(ob_function,m_update,A,b,Aeq,beq,[],[],[],opts);
    %m_restriced = m_update;
  
    %m_restriced = m_restriced/(sum(m_restriced));
     
%     W = inv(P_last);
%     R_KG = P_last * Aeq'/(Aeq*P_last*Aeq')';
%     m_restricted_intermediate = m_update-R_KG*(Aeq*m_last-beq);
%     p_restricted_intermediate = (eye(MK)-R_KG*Aeq)*P_last;
    
end