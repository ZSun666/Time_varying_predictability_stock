function[m_restricted,m_update,p_restricted] = restricted_KF_KG(Y,X,m_last,P_last,lambda,R)
    MK = size(m_last,1);
%     Y_augment = [Y;1];
%     X_augment = [X;ones(1,MK)];
%     R_augment = [R;0];
    
    m_predict = m_last;
    P_predict = P_last/lambda;
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

%     
   % m_restriced = lasso(ones(MK,MK),m_update,'Lambda',10);
   %   A = zeros(MK+1,MK);
    v = Y-X * m_predict;
    A = zeros(MK+1,MK);  b = zeros(MK+1,1);
    A(1:MK,1:MK) = -eye(MK)*v; b(1:MK) = zeros(MK,1)+m_predict;
    A(MK+1,:) = ones(1,MK)*(v);b(MK+1,:) = 1 - sum((m_predict));
%     A = zeros(2*MK+1,2*MK);  b = zeros(2*MK+1,1);
%         A(1:2*MK,:) = -eye(2*MK); b(1:2*MK) = zeros(2*MK,1);
%         A(2*MK+1,:) = ones(1,2*MK);b(2*MK+1,:) = 0.5;
% %     
%     H = eye(MK)*2; f = -2*m_update';
%     H_full = [H,-H;-H,H]; f_full = [f,-f];
%     opts = optimoptions('quadprog','Display','off','Algorithm','interior-point-convex');
%     m_restricted_full = quadprog(H_full,f_full,A,b,[],[],[],[],zeros(2*MK,1),opts);
%     m_restricted_full(m_restricted_full < 1e-5) = 0;
%     m_restricted = m_restricted_full(1:MK) - m_restricted_full(MK+1:end);
%     p_restricted = P_update;
    H = 2*eye(MK)*(X*P_predict*X'+ R); f = -2*X*P_predict;
    
    opts = optimoptions('quadprog','Display','off','Algorithm','interior-point-convex');
    K_restricted=  quadprog(H,f,A,b,[],[],[],[],[],opts);
%     K_restricted = K_restricted_full(1:MK) - K_restricted_full(MK+1:end);
%     K_restricted(abs(K_restricted)<1e-4) = 0;
    m_restricted = m_predict + K_restricted*(Y-X * m_predict);
    p_restricted = (eye(MK) - K_restricted*X)* P_predict; 
    p_restricted = (p_restricted'+p_restricted)/2;
    m_restricted(abs(m_restricted)  <1e-4) = 0;
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