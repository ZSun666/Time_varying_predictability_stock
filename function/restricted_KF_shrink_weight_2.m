function[m_restricted,m_update,p_restricted] = restricted_KF_shrink_weight_2(Y,X,m_last,P_last,lambda,R)
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

    A = zeros(MK+1,MK);  b = zeros(MK+1,1);
    A= -eye(MK); b(1:MK) = zeros(MK,1);
    A(MK+1,:) = ones(1,MK); b(MK+1) = 1;
    H = eye(MK)*2; f = -2*m_update';
%     H_full = [H,-H;-H,H]; f_full = [f,-f];
    opts = optimoptions('quadprog','Display','off','Algorithm','interior-point-convex');
    m_restricted = quadprog(H,f,A,b,[],[],[],[],m_update,opts);
    
    m_restricted(abs(m_restricted) < 1e-5) = 0;
    
    p_restricted = P_update;
    

    

end