function [beta_draw_save,beta_mean_save,beta_var_save,sigma_save,W_draw_save,W_SSE_2_save,tau_save,lambda_save]...
            = pre_estimate_MCMC_5(Y_train,X_train,N,M,K)
MK = M*K;
Ttrain = length(Y_train);
k_Q = 0.01;
k_W = 0.01;

nburn = N*0.1;
N = N+nburn;

beta_mean_init = zeros(MK,1);
beta_var_init = eye(MK);

Q_prmean = ((k_Q)^2)*2*eye(MK)*4;     
Q_prvar = 2;

W_prmean = ((k_W)^2*2); 
W_prvar = 2;
sigma_prmean    = zeros(M,1);                % log(sigma_0) ~ N(log(sigma_OLS),I_n)
sigma_prvar     = eye(M);

% hyperparameters for beta
tau_init =10e4*ones(MK,Ttrain);
a_tau = 1;
lambda_init = ones(Ttrain,1);
e_1 = 0.001;
e_2 = 0.001;

g_0 = 1;
h_0 = 12;
c_0 = 100;
d_0 = 1;
c_bar = (0.01)^2;

%X_f_tilde = zeros(MK*K_state,N-nburn,Ttrain);


W_draw_save = zeros(M,N-nburn);

tau_save = zeros(MK,N-nburn);

lambda_save = zeros(N-nburn,1);

W_SSE_2_save = zeros(1,N-nburn);
%% ===============================================================
%inv_Qdraw = repmat(eye(MK),[1,1,Ttrain]);
inv_Qdraw = ones(MK,Ttrain);
Wdraw = eye(M);
tau_t = tau_init;
inv_VB_V = 10e-4*ones(MK,Ttrain);

lambda_t = lambda_init;


Zs          = ones(Ttrain,1);
q_s     = [  0.00730;  0.10556;  0.00002; 0.04395; 0.34001; 0.24566;  0.25750]; % probabilities
m_s     = [-10.12999; -3.97281; -8.56686; 2.77786; 0.61942; 1.79518; -1.08819]; % means
u2_s    = [  5.79596;  2.61369;  5.17950; 0.16735; 0.64009; 0.34023;  1.26261]; % variances


F_tilde = repmat(eye(MK),[1,1,Ttrain]);

beta_mean_save = zeros(MK,N-nburn);
beta_var_save = zeros(MK,MK,N-nburn);
beta_draw_save = zeros(MK,N-nburn);
sigma_save = zeros(M,N-nburn);

beta_predict_mean = zeros(MK,Ttrain);
beta_predict_var = repmat(eye(MK),[1,1,Ttrain]);
beta_update_mean = zeros(MK,Ttrain);
beta_update_var = repmat(eye(MK),[1,1,Ttrain]);

beta_draw = zeros(MK,Ttrain);
sigma_draw = zeros(1,Ttrain);


for n = 1:N
    
 
     
    for t = 1:Ttrain        
        
        if t == 1
            beta_mean_last = beta_mean_init;
            beta_var_last = beta_var_init;
        else
            beta_mean_last = beta_update_mean(:,t-1);
            beta_var_last = beta_update_var(:,:,t-1);
        end
        %VB_Q_tilde(:,:,t) = inv(inv(VB_Q(:,:,t)));
        Q_tilde = diag(1./(inv_Qdraw(:,t)+inv_VB_V(:,t)));
        F_tilde(:,:,t) = Q_tilde*diag(inv_Qdraw(:,t));
        beta_predict_mean(:,t) = F_tilde(:,:,t)  * beta_mean_last;
        beta_predict_var(:,:,t) = F_tilde(:,:,t) * beta_var_last * F_tilde(:,:,t)'+ Q_tilde;
        % update step
        f_temp = (X_train(t,:)*beta_predict_var(:,:,t)*X_train(t,:)'+exp(sigma_draw(t)));
        KG = beta_predict_var(:,:,t) *X_train(t,:)'/f_temp; % kalman gain
        
        beta_update_mean(:,t) = beta_predict_mean(:,t) + KG*(Y_train(t)-X_train(t,:) * beta_predict_mean(:,t));
        beta_update_var(:,:,t) = (eye(MK) - KG*X_train(t,:))* beta_predict_var(:,:,t);
        beta_update_var(:,:,t) = (10000*beta_update_var(:,:,t)+10000*beta_update_var(:,:,t)')/20000;
        %beta_draw(:,t) = beta_update_mean(:,t) + chol(beta_update_var(:,:,t))'*randn(MK,1);
        
%         [L,D] = ldl(beta_update_var(:,:,t));
%         d = diag(D);
%         if min(d) > 0
%         root_D = diag(sqrt(d));
%         s_temp = L * root_D; 
%         else
%         index_temp = find(d <= 0);
%         d(index_temp) = 0 ;
%         root_D = diag(sqrt(d));
%         s_temp = L * root_D;
%         end
% 
%         if isreal(s_temp) == 0
%         disp("t");
%         end
%         
%         beta_draw(:,t) = beta_update_mean(:,t) + s_temp*randn(MK,1);
%         
    end

    %     VB_beta_mean_save = beta_update_mean;
    %     VB_beta_var_save = beta_update_var;
%     beta_smooth_mean(:,t) = beta_update_mean(:,t);
%     beta_smooth_var(:,:,t) = beta_update_var(:,:,t);

    [L,D] = ldl(beta_update_var(:,:,t));
    d = diag(D);
    if min(d) > 0
    root_D = diag(sqrt(d));
    s_temp = L * root_D; 
    else
    index_temp = find(d <= 0);
    d(index_temp) = 0 ;
    root_D = diag(sqrt(d));
    s_temp = L * root_D;
    end

    if isreal(s_temp) == 0
    disp("t");
    end
% 
%     beta_draw(:,t) = beta_update_mean(:,t) + s_temp*randn(MK,1);
    beta_draw(:,t) = beta_update_mean(:,t) + s_temp*randn(MK,1);
    
   %% Backward recurssions
  for  t = Ttrain-1:-1:1
    %for i=1:t-1
        bf = beta_draw(:,t+1);
        btt = beta_update_mean(:,t);
        Vtt = (beta_update_var(:,:,t));
        %inv_f = inv(eye(MK)+Vtt\Q_tilde);
        f = Vtt + Q_tilde;
        %inv_f = inv(f);
        inv_f = Vtt/f;
        cfe = bf - btt;
        %bmean = btt + Vtt*inv_f*cfe;
        %bvar = Vtt - Vtt*inv_f*Vtt;
        bmean = btt + inv_f*cfe;
        bvar = (eye(MK) - inv_f)*Vtt;
        bvar = (bvar'*10000+bvar*10000)/20000;
        
        [L,D] = ldl(bvar);
        d = diag(D);
        if min(d) > 0
        root_D = diag(sqrt(d));
        s_temp = L * root_D; 
        else
        index_temp = find(d <= 0);
        d(index_temp) = 0 ;
        root_D = diag(sqrt(d));
        s_temp = L * root_D;
        end

        if isreal(s_temp) == 0
        disp("t");
        end
        
        beta_draw(:,t) = bmean + s_temp*randn(MK,1);
    end
% beta_smooth_mean(:,t) = beta_update_mean(:,t);
% beta_smooth_var(:,:,t) = beta_update_var(:,:,t);
%     for t = Ttrain-1:-1:1
%         C_temp = beta_update_var(:,:,t)*F_tilde(:,:,t)'/beta_predict_var(:,:,t+1);
%         beta_smooth_mean(:,t) = beta_update_mean(:,t)+C_temp*(beta_smooth_mean(:,t+1)-beta_predict_mean(:,t+1));
%         beta_smooth_var(:,:,t) = beta_update_var(:,:,t)+C_temp*(beta_smooth_var(:,:,t+1)-beta_predict_var(:,:,t+1))*C_temp';
%         
%         beta_smooth_var(:,:,t) = 0.5*(beta_smooth_var(:,:,t)'+beta_smooth_var(:,:,t));
%         
%                 [L,D] = ldl(beta_smooth_var(:,:,t));
%         d = diag(D);
%         if min(d) > 0
%         root_D = diag(sqrt(d));
%         s_temp = L * root_D; 
%         else
%         index_temp = find(d <= 0);
%         d(index_temp) = 0 ;
%         root_D = diag(sqrt(d));
%         s_temp = L * root_D;
%         end
% 
%         if isreal(s_temp) == 0
%         disp("t");
%         end
%         beta_draw(:,t) = beta_smooth_mean(:,t) + s_temp*randn(MK,1);
%     end
    
    %% draw hyperparameters
    
    Q_SSE_2 = zeros(MK,MK);
    for t = 1:Ttrain
        if t >1
            
                Q_SSE_2 = (beta_draw(:,t)-beta_draw(:,t-1))*(beta_draw(:,t)-beta_draw(:,t-1))';
            
        else
            
                Q_SSE_2 = beta_draw(:,t)*beta_draw(:,t)';
          
        end
%     Q_SSE_2 = zeros(MK,1);
%     for t = 1:Ttrain
%         for j = 1:MK
%             if t >1
% 
%                     Q_SSE_2(j) = Q_SSE_2(j) +(beta_draw(j,t)-beta_draw(j,t-1))*(beta_draw(j,t)-beta_draw(j,t-1))';
% 
%             else
% 
%                     Q_SSE_2(j) = beta_draw(j,t)*beta_draw(j,t)';
% 
%             end
%         end
%         for j = 1:MK
%                    
%             tau_t(j,t) =  gigrnd(a_tau - 0.5,a_tau*lambda_t(t),beta_draw(j,t)^2,1);
% 
% 
%             if tau_t(j,t) ==0 || isnan(tau_t(j,t)) || isinf(tau_t(j,t))
%                 tau_t(j,t) = 0.0001; %introduce some drift
%             end
%             
%             inv_Vdraw(j,t) = 1/tau_t(j,t);
%         end
%         
%         lambda_t(t) = gamrnd(e_1+a_tau*MK,1/(e_2+mean(tau_t(:,t),1)*0.5*a_tau*MK));
        for j = 1:MK
            gt(t,1)     = g_0 + 0.5;
            ht(t,j)     = h_0 + 0.5*beta_mean_save(j,t)^2;
            tau_t(j) = ht(t,j)/gt(t,1);
            c_bar = 0.0001;
            pi_0(t) = 0.1;
            l_0         = lnormpdf(beta_draw(j,t),0,sqrt(c_bar*tau_t(j) )) + 1e-20;
            l_1         = lnormpdf(beta_draw(j,t),0,sqrt(tau_t(j) )) + 1e-20;
            pi_t(j)       = 1/( 1 + ((1-pi_0(t))/pi_0(t))* exp(l_0 - l_1) );

            %             pi_t(j) = mvnpdf(VB_beta_mean_save(j,t),0,tau_t(j))*pi_0(t)...
            %                    /(mvnpdf(VB_beta_mean_save(j,t),0,tau_t(j))*pi_0(t)+...
            %                    mvnpdf(VB_beta_mean_save(j,t),0,tau_t(j)*c_bar)*(1-pi_0(t)));
            if rand <pi_t(j)
                inv_VB_V(j,t) = 1/tau_t(j) ;
            else
                inv_VB_V(j,t) = 1/(tau_t(j) *c_bar);
            end
            
           % = 1/((1-pi_t(j)) * c_bar * tau_t(j)+ pi_t(j)^2 *tau_t(j));

        end

        for j = 1:MK
            Qinv        = inv(Q_SSE_2(j,j));

            inv_Qdraw(j,t)   = gamrnd(Qinv,1);
        end

    end

%     Qinv        = inv(diag(Q_SSE_2)+ Q_prmean);
%     for j = 1:MK
%         inv_Qdraw(j,j,1:t)   = repmat(wish(Qinv(j,j),t+Q_prvar),[1,1,Ttrain]);
%     end
%   Qinv        = inv((Q_SSE_2)+ Q_prmean);
%     for j = 1:MK
%         inv_Qdraw(:,:,1:t)   = repmat(wish(Qinv,t+Q_prvar),[1,1,Ttrain]);
%     end
    %Qdraw       = inv(inv_Qdraw);            % this is a draw from Q
    
    %% draw sigma
       
        for tt = 1:Ttrain
            yhat(tt) = Y_train(tt) - X_train(tt,:) * beta_draw(:,tt);
        end
        [statedraw,yss,~]           = mcmc_draw_state([],sigma_draw,yhat,m_s,u2_s,q_s,1,Ttrain);
        [sigma_draw,~,sigt,Wdraw,W_SSE_2_draw]  = mcmc_draw_sigma(statedraw,Wdraw,yss,Zs,m_s,u2_s,1,Ttrain,sigma_prmean,sigma_prvar,W_prmean,W_prvar);

        
   
        % sample P_0

        % storage samples
        if n > nburn
            
         
           
            beta_draw_save(:,n-nburn) = beta_draw(:,end);
            beta_mean_save(:,n-nburn) = beta_update_mean(:,end);
            beta_var_save(:,:,n-nburn) = beta_update_var(:,:,end);
            sigma_save(:,n-nburn) = sigma_draw(end);
            W_draw_save (:,n-nburn) = Wdraw;
            W_SSE_2_save(:,n-nburn) = W_SSE_2_draw;
            tau_save(:,n-nburn) = tau_t(:,end);          
            lambda_save(n-nburn) = lambda_t(end);
            
        end
      
end


end


