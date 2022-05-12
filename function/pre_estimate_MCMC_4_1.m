function [beta_fix,theta_fix,f_tilde,sigma_t,W_draw,W_SSE_2,tau_save,xi_save,kappa_save,lambda_save,a_tau_save,a_xi_save,zeta_t]...
            = pre_estimate_MCMC_4_1(Y_train,X_train,N,M,K,K_state,T_pre,zeta_range,a_fix)
MK = M*K;
Ttrain = length(Y_train);
k_W = 0.01;
W_prmean = ((k_W)^2*2); 
W_prvar = 2;
nburn = N*0.1;
N = N+nburn;
zeta_t = median(zeta_range);
beta_init = zeros(MK,1);
theta_init = zeros(K*K_state,1);
sigma_init = zeros(1,Ttrain);

sigma_prmean    = zeros(M,1);                % log(sigma_0) ~ N(log(sigma_OLS),I_n)
sigma_prvar     = eye(M);

% hyperparameters for beta
tau_init = 0.1*ones(MK,1);
a_tau = 0.01;
lambda_init = 1;
P_init = ones(K_state,1);
e_1 = 0.001;
e_2 = 0.001;

% hyperparameters for theta
xi_init = 0.1*ones(MK*K_state,1);
a_xi = 0.01;
kappa_init = 1;
d_1 = 0.001;
d_2 = 0.001;

% storage matrix
beta_fix = zeros(MK,N-nburn);
theta_fix = zeros(K*K_state,N-nburn);
f_tilde = zeros(K_state,N-nburn);
%X_f_tilde = zeros(MK*K_state,N-nburn,Ttrain);

sigma_t = zeros(M,N-nburn,T_pre);
W_draw = zeros(M,N-nburn);

tau_save = zeros(MK,N-nburn);
xi_save =  zeros(MK*K_state,N-nburn);
kappa_save = zeros(N-nburn,1);
lambda_save = zeros(N-nburn,1);
a_tau_save = zeros(N-nburn,1);
a_xi_save = zeros(N-nburn,1);
W_SSE_2 = zeros(1,N-nburn);
%% ===============================================================
beta_draw = beta_init;
theta_draw = theta_init;
sigma_draw = sigma_init;
Wdraw = eye(M);
alpha_draw = [beta_draw;theta_draw];
f_tilde_draw = zeros(K_state,Ttrain);
H_t = ones(Ttrain,1).*exp(sigma_draw');
P_draw = P_init;

tau_draw = tau_init;
xi_draw = xi_init;

lambda_draw = lambda_init;
kappa_draw = kappa_init;

Zs          = ones(Ttrain,1);
q_s     = [  0.00730;  0.10556;  0.00002; 0.04395; 0.34001; 0.24566;  0.25750]; % probabilities
m_s     = [-10.12999; -3.97281; -8.56686; 2.77786; 0.61942; 1.79518; -1.08819]; % means
u2_s    = [  5.79596;  2.61369;  5.17950; 0.16735; 0.64009; 0.34023;  1.26261]; % variances
timer_k_mean = 0;
for n = 1:N
    
    % draw beta_tilde
     y_star =  Y_train- X_train * beta_draw;
    % x_star = X_train .* theta_draw';
     x_star = X_train * reshape(theta_draw,[MK,K_state]);
     
    [ftdrawc,~] = carter_kohn(y_star',x_star,H_t,eye(K_state),K_state,1,Ttrain,zeros(K_state,1),diag(P_draw));
    %Btdrawc = AWOL(y_star',x_star,H_t,diag(P_draw));
    f_tilde_draw = ftdrawc';
        %f_tilde(:,nn,n,:) = f_tilde_draw(:,nn,:);
    
    %% divided beta&theta into active/inactive set
       
    for tt = 1:Ttrain
        X_f_train(tt,:) = kron(f_tilde_draw(tt,:),X_train(tt,:));
    end
        
        % draw beta and theta jointly
        Z_t = [X_train,X_f_train];
    %% draw beta/theta fix
    %Z_squared_sigma = Z_t'*H_temp*Z_t;
    tau_xi = [tau_draw;xi_draw];
    %Z_squared = Z_t'*Z_t;
     % draw beta and theta jointly
    Z_t_tilde = Z_t' * (eye(Ttrain)*1./exp(sigma_draw'));
    a_prior = diag([tau_xi]); % A_0
    root_a_prior = sqrt(a_prior); %A_0^(1/2)
    A_temp = (root_a_prior * Z_t_tilde * Z_t * root_a_prior+eye(MK+MK*K_state));       % A_t star
    a_var = (root_a_prior / A_temp) * root_a_prior;
    a_mean = a_var * Z_t_tilde * Y_train;
    
    [L,D] = ldl(a_var);
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

    alpha_draw = a_mean + s_temp*randn(MK+MK*K_state,1);

   
    
    beta_draw = alpha_draw(1:MK);
    theta_draw = alpha_draw(MK+1:end);
    %% draw hyperparameters
        if ~a_fix == 1
            a_xi_old = a_xi;
            log_a_xi_old = log(a_xi_old);
            accept_rate_xi_temp = 1;
            a_xi_prop = exp(mvnrnd(log_a_xi_old,0.1));
            for j = 1:MK*K_state         
            accept_rate_xi_temp = accept_rate_xi_temp...
                          * (sqrt(a_xi_prop*kappa_draw)^(a_xi_prop+0.5))/(sqrt(a_xi_old*kappa_draw)^(a_xi_old+0.5))...
                          * abs(theta_draw(j))^(a_xi_prop-a_xi_old) * 2^(a_xi_old-a_xi_prop)...
                          * gamma(a_xi_old)/gamma(a_xi_prop)...
                          * besselk(a_xi_prop-0.5,sqrt(a_xi_prop*kappa_draw)*abs(theta_draw(j)))...
                          / besselk(a_xi_old-0.5,sqrt(a_xi_old*kappa_draw)*abs(theta_draw(j)));

            end
            accept_rate_xi_temp = accept_rate_xi_temp *exppdf(a_xi_prop,10)/exppdf(a_xi_old,10)...
                                  * a_xi_prop/a_xi_old;
            accept_xi = rand;
            if accept_xi < accept_rate_xi_temp
                a_xi = a_xi_prop;
            end


            a_tau_old = a_tau;
            log_a_tau_old = log(a_tau_old);
            accept_rate_tau_temp = 1;
            a_tau_prop = exp(mvnrnd(log_a_tau_old,0.1));
            for j = 1:MK         
                accept_rate_tau_temp = accept_rate_tau_temp...
                  * (sqrt(a_tau_prop*lambda_draw)^(a_tau_prop+0.5))/(sqrt(a_tau_old*lambda_draw)^(a_tau_old+0.5))...
                  * abs(beta_draw(j))^(a_tau_prop-a_tau_old) * 2^(a_tau_old-a_tau_prop)...
                  * gamma(a_tau_old)/gamma(a_tau_prop)...
                  * besselk(a_tau_prop-0.5,sqrt(a_tau_prop*lambda_draw)*abs(beta_draw(j)))...
                  / besselk(a_tau_old-0.5,sqrt(a_tau_old*lambda_draw)*abs(beta_draw(j)));

            end
            accept_rate_tau_temp = accept_rate_tau_temp *exppdf(a_tau_prop,10)/exppdf(a_tau_old,10)...
                      * a_tau_prop/a_tau_old;
            accept_tau = rand;
            if accept_tau < accept_rate_tau_temp
                a_tau = a_tau_prop;
            end
        end
        for j = 1:MK
                tau_draw(j) =  gigrnd(a_tau - 0.5,a_tau*lambda_draw,beta_draw(j)^2,1);

            if tau_draw(j) == 0 || isnan(tau_draw(j))  || isinf(tau_draw(j))
                tau_draw(j) = 0.000000001;
            end
        end
        

        
        for j = 1:MK*K_state
            %if abs(theta_draw(j)) > 0.0001
                xi_draw(j) = gigrnd(a_xi - 0.5,a_xi*kappa_draw,theta_draw(j)^2,1);
            %else
%                 if (a_xi -0.5) > 0 
%                     xi_draw(j) = gamrnd(a_xi - 0.5,2/(a_xi*kappa_draw));
%                 else
%                     xi_draw(j) = 1/gamrnd(0.5-a_xi,2/(a_xi*kappa_draw));
%                 end
            %end
             if xi_draw(j) == 0 || isnan(xi_draw(j))  || isinf(xi_draw(j))
                xi_draw(j) = 0.0000000001;
            end
        end
        % sample lambda&kappa
        lambda_draw = gamrnd(e_1+a_tau*MK,1/(e_2+mean(tau_draw)*0.5*a_tau*MK));
        kappa_draw = gamrnd(d_1+a_xi*(MK*K_state),1/(d_2+mean(xi_draw)*0.5*a_xi*(MK*K_state)));
                % 2-mean 

        beta_t = beta_draw + reshape(theta_draw,[MK,K_state])*f_tilde_draw(end,:)';
        X_beta_t = X_train(end,:)'.*beta_t;
        
        %big_index = 1:MK;
        for ii = 1:length(zeta_range)
            zeta_t_select = zeta_range(ii);
            small_index = 1:MK;
            diff_mM = 10000;
            small_index_length = MK;
            
            while diff_mM >zeta_t_select && small_index_length > 1
                signal_temp = abs(X_beta_t(small_index));
                
                cluster_index = kmeans(signal_temp,2);
                mean_temp = [mean(signal_temp(find(cluster_index==1))),mean(signal_temp(find(cluster_index==2)))];
                [big_m,i_big] = max(mean_temp);
                [small_m,i_small] = min(mean_temp);

                
                big_index = small_index(find(cluster_index == i_big));
                small_index = small_index(find(cluster_index == i_small));
                small_index_length = length(small_index);
                diff_mM = abs(big_m-small_m);
            end
                if n >nburn
                    num_signal(n-nburn,ii) = length(small_index);
                end
         
        end

        % sample sigma
        % Posterior of SIGMA|ALPHA,Data ~ iW(inv(S_post),v_post)
       
        for tt = 1:Ttrain
            yhat(tt) = Y_train(tt) - X_train(tt,:) * beta_draw -X_train(tt,:) *reshape(theta_draw,[MK,K_state])* f_tilde_draw(tt,:)';
        end
        [statedraw,yss,~]           = mcmc_draw_state([],sigma_draw,yhat,m_s,u2_s,q_s,1,Ttrain);
        [sigma_draw,~,sigt,Wdraw,W_SSE_2_draw]  = mcmc_draw_sigma(statedraw,Wdraw,yss,Zs,m_s,u2_s,1,Ttrain,sigma_prmean,sigma_prvar,W_prmean,W_prvar);


        H_t = sigt.^2;
        % sample P_0

        % storage samples
        if n > nburn
            
            beta_fix(:,n-nburn) = beta_draw;
            theta_fix(:,n-nburn) = theta_draw;
            f_tilde(:,n-nburn) = f_tilde_draw(end,:);
            %X_f_tilde(:,n-nburn,:) = X_f_train';
            sigma_t(:,n-nburn,:) = sigma_draw;
            W_draw (:,n-nburn) = Wdraw;
            W_SSE_2(:,n-nburn) = W_SSE_2_draw;
            tau_save(:,n-nburn) = tau_draw;
            xi_save(:,n-nburn) = xi_draw;
            lambda_save(n-nburn) = lambda_draw;
            kappa_save(n-nburn) = kappa_draw;
            a_tau_save(n-nburn) = a_tau;
            a_xi_save(n-nburn) = a_xi;
        end
      
end
    for ii = 1:length(zeta_range)
    mode_num_signal(ii) = MK-mode(num_signal(:,ii));
    end
    change_temp = mode_num_signal(2:ii)-mode_num_signal(1:ii-1);
    [~,i_temp] = max(change_temp);
    zeta_t = zeta_range(i_temp);

end


