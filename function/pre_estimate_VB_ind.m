function[VB_beta_mean_save,VB_beta_var_save,VB_sigma_save,Q_SSE_2,VB_Q_tilde_save,VB_F_tilde_save]...
        = pre_estimate_VB_ind(Y_pre,X_pre)
    
    Lpre = length(Y_pre);
    X_pre = [ones(Lpre,1),X_pre];
    MK = size(X_pre,2);
    inv_VB_Q = [ones(MK,Lpre)];
    inv_VB_V = [zeros(MK,Lpre)];
    VB_sigma_save = 0.1*ones(Lpre,1);
    F_new = 0;
    F_old = 1;
    Threshold = 0.01;
    niter = 0;
    
    pi_0 = 0.1*ones(Lpre,1);
    g_0 = 1;
    h_0 = 12;
    c_0 = Lpre;
    d_0 = 1;
    c_bar = 0.0001;
    %hyper for sigma
    delta = 0.9;
    a_0 = 0.01;
    b_0 = 0.01;
    %% Forward Filter
    while niter < 100 && abs((F_new - F_old)) > Threshold
    for t = 1:Lpre
    if t == 1
        beta_mean_last = zeros(MK,1);
        beta_var_last = 4*eye(MK);
    else
        beta_mean_last = beta_update_mean(:,t-1);
        beta_var_last = beta_update_var(:,:,t-1);
    end
    %VB_Q_tilde(:,:,t) = inv(inv(VB_Q(:,:,t)));
    VB_Q_tilde(:,:,t) = inv(diag(inv_VB_Q(:,t))+diag(inv_VB_V(:,t)));
    VB_F_tilde(:,:,t) = VB_Q_tilde(:,:,t)*diag(inv_VB_Q(:,t));
    beta_predict_mean(:,t) = VB_F_tilde(:,:,t)  * beta_mean_last;
    beta_predict_var(:,:,t) = VB_F_tilde(:,:,t) * beta_var_last * VB_F_tilde(:,:,t)'+ VB_Q_tilde(:,:,t);
    % update step
    f_temp = (X_pre(t,:)*beta_predict_var(:,:,t)*X_pre(t,:)'+VB_sigma_save(t));
    KG = beta_predict_var(:,:,t) *X_pre(t,:)'/f_temp; % kalman gain
    beta_update_mean(:,t) = beta_predict_mean(:,t) + KG*(Y_pre(t)-X_pre(t,:) * beta_predict_mean(:,t));
    beta_update_var(:,:,t) = (eye(MK) - KG*X_pre(t,:))* beta_predict_var(:,:,t);

    end

    %     VB_beta_mean_save = beta_update_mean;
    %     VB_beta_var_save = beta_update_var;
    VB_beta_mean_save(:,t) = beta_update_mean(:,t);
    VB_beta_var_save(:,:,t) = beta_update_var(:,:,t);


    %% Backward Smooth
    for t = Lpre-1:-1:1
    C_temp = beta_update_var(:,:,t)*VB_F_tilde(:,:,t)/beta_predict_var(:,:,t+1);
    VB_beta_mean_save(:,t) = beta_update_mean(:,t)+C_temp*(VB_beta_mean_save(:,t+1)-beta_predict_mean(:,t+1));
    VB_beta_var_save(:,:,t) = beta_update_var(:,:,t)+C_temp*(VB_beta_var_save(:,:,t+1)-beta_predict_var(:,:,t+1))*C_temp';

    end

    %% Posteiror of beta-related parameters
    a_last = a_0;
    b_last = b_0;
    Q_SSE_2 = 0;
    for t = 1:Lpre
        if t >1
            Q_SSE_2 = Q_SSE_2 + (VB_beta_mean_save(:,t)-VB_beta_mean_save(:,t-1))*(VB_beta_mean_save(:,t)-VB_beta_mean_save(:,t-1))';
        else
            Q_SSE_2 = VB_beta_mean_save(:,t)*VB_beta_mean_save(:,t)';
        end
    %             D_temp = VB_beta_var_save(:,:,t)+VB_beta_mean_save(:,t)*VB_beta_mean_save(:,t)'...
    %             +(VB_beta_var_save(:,:,t-1)+VB_beta_mean_save(:,t-1)*VB_beta_mean_save(:,t-1)')...
    %             *(eye(MK)-2*VB_F_tilde(:,:,t))';
    

    R_temp = (Y_pre(t) - X_pre(t,:)*VB_beta_mean_save(:,t))^2+X_pre(t,:)*VB_beta_var_save(:,:,t)*X_pre(t,:)';

    for j = 1:MK
        gt(t,1)     = g_0 + 0.5;
        ht(t,j)     = h_0 + 0.5*VB_beta_mean_save(j,t)^2;
        tau_t(j) = ht(t,j)/gt(t,1);

        l_0         = lnormpdf(VB_beta_mean_save(j,t),0,sqrt(c_bar*tau_t(j))) + 1e-20;
        l_1         = lnormpdf(VB_beta_mean_save(j,t),0,sqrt(tau_t(j))) + 1e-20;
        pi_t(j)       = 1/( 1 + ((1-pi_0(t))/pi_0(t))* exp(l_0 - l_1) );

    %             pi_t(j) = mvnpdf(VB_beta_mean_save(j,t),0,tau_t(j))*pi_0(t)...
    %                    /(mvnpdf(VB_beta_mean_save(j,t),0,tau_t(j))*pi_0(t)+...
    %                    mvnpdf(VB_beta_mean_save(j,t),0,tau_t(j)*c_bar)*(1-pi_0(t)));
        inv_VB_V(j,t) = 1/((1-pi_t(j)) * c_bar * tau_t(j)+ pi_t(j)^2 *tau_t(j));
        ct(t,1) = t+0.5;
        dt(j) = (d_0+ Q_SSE_2(j,j));

        %             ct(t,1) =c_0+0.5;
        %             dt(t,j) = d_0 + D_temp(j,j)/2;
        inv_VB_Q(j,t) = ct(t)/dt(j);
    end

    pi_0(t) = (1+ sum(pi_t(j)))/(2+MK);
    phi_hat(t) = (delta*a_last+0.5)/(delta * b_last+0.5*R_temp);
    a_last = a_last * delta+0.5;
    b_last = R_temp * 0.5+delta*b_last;


    end

    phi_tilde(t) = phi_hat(t);
    for t = Lpre-1:-1:1
    phi_tilde(t) = (1-delta)*phi_hat(t)+delta*phi_tilde(t+1);
    end
    VB_sigma_save = 1./phi_tilde;
    F_old = F_new;
    F_new = norm(lnormpdf(Y_pre,sum(X_pre.*VB_beta_mean_save',2),VB_sigma_save)) - sum(klgamma(gt,ht,g_0,h_0)) - sum(klgamma(ct,dt,c_0,d_0));
    niter = niter + 1 ;
    end
    
    VB_Q_tilde_save = diag(VB_Q_tilde(:,:,end));
    VB_F_tilde_save = diag(VB_F_tilde(:,:,end));
end