tic
%% 
%% model setup
date_start = [1927,01]; % start of training sample
test_start = [1988,01]; % start of test sample

date_end = [2019,12];

N = 5000; % number of particle

L_train = 60; 
test_start = datetime([test_start,1]);
%% data load
addpath('result'); addpath('function'); addpath('data');
data_raw = readtable("new_dataset_full.xlsx");

var_name = data_raw.Properties.VariableNames;

y_date = datetime(num2str(data_raw{:,1}),'InputFormat','yyyyMM');

y_all = data_raw{:,2}*100;
date_start = datetime([date_start,1]);
date_end = datetime([date_end,1]);
date_start_index = find(date_start == y_date);
date_end_index = find(date_end == y_date);
y_all = y_all(date_start_index:date_end_index);
y_date = y_date(date_start_index:date_end_index);
T_total = length(y_all);
L_test = find(y_date == test_start);
L_pre_end = L_test - L_train;
%% load predictions from first stage
load result/first_stage.mat;

L_first_stage = length(First_stage);
for t_fs = 1:L_first_stage
    x_date(t_fs) = First_stage(t_fs).date;
    x_all(t_fs,:) = First_stage(t_fs).y_predict;%+First_stage(t_fs).y_mean;
    nan_index = find(isnan(x_all(t_fs,:)));
    x_all(t_fs,nan_index) = First_stage(t_fs).y_mean;
    R2_fs(t_fs,:) = First_stage(t_fs).R2;
    
end
MK = size(x_all,2);

%% setup hyperparameters

% hyper for DVS
g_0 = 1;
h_0 = 6;
c_bar = (1e-6); 
drift = 0.01;

% hyper for variations in coefficients
k_Q = 0.1;
Q_prvar = 100;
Q_prmean = ((k_Q)^2)*(Q_prvar+1)*eye(MK)*0.1; 

%  hyper for variations in volatility
k_W = 0.1;
W_prmean = k_W^2*2;
W_prvar = 2;


%% initialize parameters by a pre-sample

date_pre_start = y_date(L_pre_end+1);
date_pre_end = y_date(L_pre_end+L_train);
y_fs_index_pre_start = find(date_pre_start == x_date);
y_fs_index_pre_end = find(date_pre_end == x_date);

y_pre = y_all(L_pre_end-120:L_pre_end-1);


R2_pre = R2_fs(y_fs_index_pre_end,:);
R2_pre(R2_pre <0.002) = 0;
R2_pre(isnan(R2_pre)) = 0 ;

non_zero_index = find(R2_pre >0.00);
omega_pre = zeros(MK,1);
omega_pre(non_zero_index) = 1./length(non_zero_index);

sigma_pre = var(y_pre);

%% temp matrix in alorithm
% store coefficients
state_omega_mean = repmat(omega_pre,[1,N]);
state_omega_norm = zeros(MK,N);
state_omega_equal = zeros(MK,N);
state_pi = zeros(MK,N);
state_omega_var = repmat(eye(MK)*0.1,[1,1,N]);
state_sigma_t = repmat(log(sigma_pre),[N,1]);


% store predictions
y_predict = zeros(N,1);
y_predict_norm = zeros(N,1);
y_predict_equal = zeros(N,1);

% store hyperparameters
Qsse_2 = zeros(MK,N);
Qdraw_inv = repmat(ones(MK,1)*1e5,[1,N]);
Q_count_t = zeros(MK,N);
Wsse_2 = zeros(N,1);
Wdraw = 0.0001*ones(N,1);

% store importance weights
normal_weight = ones(N,1)./N;
weight_t = zeros(N,1);

RSS = 0;
RSS2 = 0;
RSS3 = 0;
TSS = 0;

Result_DSC = struct;

%% main algorithm
for t =  L_pre_end+1:T_total - 1
    tt = t-L_pre_end;
    y_date_t = y_date(t,:);
    FS_index = find(y_date_t == x_date);
    R2_ind_FS = R2_fs(FS_index,:);
    
    
    y_mean = mean(y_all(1:t));
    
    y_fit_fs_t(tt,:) = x_all(FS_index-1,:) - y_mean; % the predictors
    y_predict_fs_t(tt,:) = x_all(FS_index,:) - y_mean; 
    y_t(tt) = y_all(t) - y_mean; % the target
    
    index_num = find(~isnan(y_fit_fs_t(tt,:)));
    index_num = intersect(index_num,find(~isnan(y_predict_fs_t(tt,:))));
    MK_num = length(index_num); % num of available obs
    
    
    y_true = y_all(t+1);
    
    num_VB_iter = zeros(N,1);
    %% particle filter
    parfor n = 1:N
        % temp vectors for parallel computation
        partemp_inv_VB_Q = Qdraw_inv(:,n);
        partemp_omega_mean_last_temp = state_omega_mean(:,n);
        partemp_omega_mean_last = zeros(MK,1);
        partemp_omega_mean_last(index_num) = partemp_omega_mean_last_temp(index_num);
        partemp_omega_norm = zeros(MK,1);
        partemp_omega_equal = zeros(MK,1);
        partemp_omega_mean = partemp_omega_mean_last;
        partemp_omega_var = state_omega_var(:,:,n);
        partemp_omega_var_last = partemp_omega_var;
        
        % sample the volatility
        y_hat = log((y_t(tt) - y_fit_fs_t(tt,index_num) * partemp_omega_mean_last(index_num))^2+0.0001);
        [partemp_sigma,lik] = SMC_RB_draw_sigma_3(y_hat,1,state_sigma_t(n),0,Wdraw(n));
        Wsse_2(n) = Wsse_2(n) + (partemp_sigma-state_sigma_t(n))^2;
        Winv        = inv(Wsse_2(n) + W_prmean);
        Winvdraw    = wish(Winv,tt + W_prvar);
        partemp_W_draw       = inv(Winvdraw); 
        
        % temp vector for VB
        partemp_inv_VB_V = zeros(MK,1);
        D_temp = zeros(MK,1);
        pi_t = zeros(MK,1);
        partemp_pi_0 = 0.1;
        F_new = 0;
        F_old = 1;
        Threshold = 0.0001;
        
        
       
        
        partemp_y_mean = y_fit_fs_t(tt,index_num)*partemp_omega_mean_last(index_num);
        partemp_y_var = y_fit_fs_t(tt,index_num)*(partemp_omega_var(index_num,index_num))*y_fit_fs_t(tt,index_num)'+exp(partemp_sigma);
        if partemp_y_var >0
            weight_t(n) = (mvnpdf(y_t(tt),partemp_y_mean,partemp_y_var)*mvnpdf(partemp_sigma,state_sigma_t(n),Wdraw(n))/lik)*normal_weight(n);
        else
            weight_t(n) = 0;
        end
        
        for m = index_num
            if partemp_omega_var_last(m,m) < drift
                partemp_omega_var_last(m,m) = drift;
            end
        end
        %% VB algorithm
        while num_VB_iter(n) < 100 && abs((F_new - F_old)) > Threshold
            VB_Q_tilde = (1./(partemp_inv_VB_Q(index_num)+partemp_inv_VB_V(index_num)));
            
            VB_F_tilde = (VB_Q_tilde.*(partemp_inv_VB_Q(index_num)));
            VB_F_tilde_matrix = diag(VB_F_tilde);
    
            % prediction step
            omega_predict_mean = VB_F_tilde.* partemp_omega_mean_last(index_num);
            omega_predict_var = VB_F_tilde_matrix * partemp_omega_var_last(index_num,index_num) *VB_F_tilde_matrix'+ diag(VB_Q_tilde);
          
            % update step
            f_temp = (y_fit_fs_t(tt,index_num) * omega_predict_var* y_fit_fs_t(tt,index_num)'+exp(partemp_sigma));
            KG = omega_predict_var *y_fit_fs_t(tt,index_num)'/f_temp; % kalman gain
            v = y_t(tt)-y_fit_fs_t(tt,index_num) * omega_predict_mean;
            partemp_omega_mean(index_num) = omega_predict_mean + KG*(v);
            partemp_omega_var(index_num,index_num) = (eye(MK_num) - KG*y_fit_fs_t(tt,index_num))* omega_predict_var;
            
            
            %% posterior for Variable selection
            
            for j = index_num
                gt  = g_0 + 0.5;
                ht     = h_0 + 0.5*partemp_omega_mean(j)^2;
                partemp_tau_t = ht/gt;
    
                l_0         = lnormpdf(partemp_omega_mean(j),0,(c_bar*partemp_tau_t)) + 1e-20;
                l_1         = lnormpdf(partemp_omega_mean(j),0,(partemp_tau_t)) + 1e-20;
                pi_t(j)      = 1/( 1 + ((1-partemp_pi_0)/partemp_pi_0)* exp(l_0 - l_1));
                partemp_inv_VB_V(j) = 1/((1-pi_t(j))^2 * c_bar * partemp_tau_t+ pi_t(j)^2 *partemp_tau_t);
              
            end
          
            partemp_pi_0 = (1+sum(pi_t))/(2+MK_num);
            num_VB_iter(n) = num_VB_iter(n)+1;
            F_old = F_new;
            F_new = (lnormpdf(y_t(tt),y_fit_fs_t(tt,index_num)*partemp_omega_mean(index_num),y_fit_fs_t(tt,index_num)*partemp_omega_var(index_num,index_num)*y_fit_fs_t(tt,index_num)'+exp(partemp_sigma)));%- sum(klgamma(gt,ht,g_0,h_0)) - sum(klgamma(c_t,d_t,c_0,d_0));
        end
        
        
        %% draw Q
        
        Qsse_temp = Qsse_2(:,n) + (partemp_omega_mean-partemp_omega_mean_last).^2;
        Qsse_2(:,n)  = Qsse_temp;
        Qinv = zeros(MK,1);
        Qinv_draw_temp = zeros(MK,1);
        tt_temp = Q_count_t(:,n);
        for k = 1:MK
            if pi_t(k) > 0.5
                tt_temp(k) = tt_temp(k)+1;
            end
            Qinv(k)        = inv((Qsse_temp(k)) + Q_prmean(k,k));
            Qinv_draw_temp(k)    = diag(wish(Qinv(k),tt_temp(k) + Q_prvar));
        end
        Qdraw_inv(:,n) = Qinv_draw_temp;
        Q_count_t(:,n) = tt_temp;
        
        %% restrict weights non-negative
        A= -eye(MK_num); b = zeros(MK_num,1);
        Aeq = ones(1,MK_num);beq = 1;
        beq = beq - Aeq *omega_predict_mean;
        Aeq = Aeq*v;
        b = b-A*omega_predict_mean;
        A = A*v;
        

        H = 2*(y_fit_fs_t(tt,index_num)*omega_predict_var*y_fit_fs_t(tt,index_num)'+ exp(partemp_sigma))*eye(MK_num);
        f = -2*y_fit_fs_t(tt,index_num)*omega_predict_var;
        % quadratic programming
        opts = optimoptions('quadprog','Display','off','Algorithm','interior-point-convex');
        KG_restricted = quadprog(H,f,A,b,[],[],[],[],KG,opts);
        partemp_omega_mean(index_num) = omega_predict_mean + KG_restricted*(v);
        partemp_omega_var(index_num,index_num) = (eye(MK_num) - KG_restricted*y_fit_fs_t(tt,index_num))* omega_predict_var;
    
  
        non_zero_index = find(partemp_omega_mean > 0.001);
        total_index = intersect(non_zero_index,index_num);
        partemp_omega_norm(total_index) = partemp_omega_mean(total_index);
        if sum(partemp_omega_norm) >0 
            partemp_omega_norm = partemp_omega_norm./sum(partemp_omega_norm);
        end
      
        partemp_omega_equal(total_index) = 1./length(total_index);
  
    
        y_predict_mean = y_predict_fs_t(tt,index_num)*partemp_omega_mean(index_num);
        y_predict_mean_norm = y_predict_fs_t(tt,index_num)*partemp_omega_norm(index_num);
        y_predict_mean_equal = y_predict_fs_t(tt,index_num)*partemp_omega_equal(index_num);
        y_predict_var = abs(y_predict_fs_t(tt,index_num) * partemp_omega_var(index_num,index_num)*y_predict_fs_t(tt,index_num)')+exp(partemp_sigma);
        y_predict(n) = y_predict_mean;
       

        state_sigma_t(n) = partemp_sigma;
        state_omega_mean(:,n) = partemp_omega_mean;
        state_omega_norm(:,n) = partemp_omega_norm;
        state_omega_equal(:,n) = partemp_omega_equal;
        state_omega_var(:,:,n) = partemp_omega_var;
        state_pi(:,n) = pi_t;
        Wdraw(n) = partemp_W_draw;
    %         Qdraw(:,:,n) = partemp_Q_draw;
    end
    normal_weight = weight_t./sum(weight_t);
    ESS(tt) = inv(normal_weight'* normal_weight); 

    %% resample
    if ESS(tt) < 0.5*N
        resample_index = randsample(N,N,true,normal_weight);
        normal_weight = ones(N,1)./N;
        state_omega_mean = state_omega_mean(:,resample_index);
        state_omega_norm = state_omega_norm(:,resample_index);
        state_omega_equal = state_omega_equal(:,resample_index);
        state_omega_var = state_omega_var(:,:,resample_index);
        state_sigma_t = state_sigma_t(resample_index);
        state_sigma_t = state_sigma_t(resample_index);
        y_predict = y_predict(resample_index);
        
        state_pi = state_pi(:,resample_index);
        Qdraw_inv = Qdraw_inv(:,resample_index);
        Qsse_2 = Qsse_2(:,resample_index);
        Q_count_t = Q_count_t(:,resample_index);
        
        Wdraw = Wdraw(resample_index);
        Wsse_2 = Wsse_2(resample_index);
    
    end
    %% storage
    for m =1:MK
        state_omega_save(m,tt) = weighted_mean(state_omega_mean(m,:),normal_weight);
        state_omega_norm_save(m,tt) = weighted_mean(state_omega_norm(m,:),normal_weight);
        state_omega_equal_save(m,tt) = weighted_mean(state_omega_equal(m,:),normal_weight);
        state_pi_save(m,tt) = weighted_mean(state_pi(m,:),normal_weight);
    end
    
    non_zero_index = find(abs(state_omega_save(:,tt)) > 1e-2);
    total_index = intersect(non_zero_index,index_num);
    state_omega_norm_temp = zeros(MK,1);
    
    state_omega_norm_temp(total_index) = max(0,state_omega_save(total_index,tt)./sum(state_omega_save(total_index,tt)));
    state_omega_norm_save(:,tt) = state_omega_norm_temp;
    
    state_omega_equal_temp = zeros(MK,1);
    state_omega_equal_temp(total_index) = 1./length(total_index);
    
    state_omega_equal_save(:,tt) = state_omega_equal_temp;
    
    
    state_sigma_save(tt) = weighted_mean(state_sigma_t,normal_weight);
    y_predict_save(tt) = weighted_mean(y_predict,normal_weight) + y_mean;
    y_predict_save_norm(tt) = y_predict_fs_t(tt,index_num)*state_omega_norm_save(index_num,tt) + y_mean;
    y_predict_save_equal(tt) = y_predict_fs_t(tt,index_num)*state_omega_equal_save(index_num,tt) + y_mean;
  
    if tt > L_train
        RSS = RSS + (y_true - y_predict_save(tt))^2;  
        RSS2 = RSS2 + (y_true - y_predict_save_norm(tt))^2;  
        RSS3 = RSS3 + (y_true - y_predict_save_equal(tt))^2;  
        TSS = TSS + (y_true - mean(y_all(1:t)))^2;
        R2(tt) = 1-RSS/TSS;  
        R2_2(tt) = 1-RSS2/TSS;
        R2_3(tt) = 1-RSS3/TSS;
    end
    Result_DSC(tt).x_predict = y_predict_fs_t(tt,:);
    Result_DSC(tt).R2_FS = R2_ind_FS;
    Result_DSC(tt).date = y_date_t;

    Result_DSC(tt).y_predict = y_predict_save(tt);
    Result_DSC(tt).y_predict_norm = y_predict_save_norm(tt);
    Result_DSC(tt).y_predict_equal = y_predict_save_equal(tt);
    Result_DSC(tt).y_to_t =y_all(1:t);
    Result_DSC(tt).ESS = ESS(tt);
    Result_DSC(tt).y_mean = y_mean;
    Result_DSC(tt).pi = state_pi_save(:,tt);
    Result_DSC(tt).y_true = y_true;
    Result_DSC(tt).beta = state_omega_save(:,tt);
    
    Result_DSC(tt).beta_norm = state_omega_norm_save(:,tt);
   
    Result_DSC(tt).beta_equal = state_omega_equal_save(:,tt);

    Result_DSC(tt).sigma = state_sigma_save(tt);
    toc
    disp([num2str((tt/(T_total-L_test))*100),'%']);
end
% Result_tvc(1).hyperpara = [g_0,h_0,c_bar,N,drift,L_train,Q_prvar];
save('result/result_DSC_c_copy','Result_DSC')
