%% this code is to replicate the forecasts result by Combination ENet (CEnet)
tic
%% model setup
date_start = [1948,01]; %YY,MM
date_end = [2019,12]; %YY,MM
test_start = [1988,01]; %YY,MM
estimate_window =0; % length of estimate window, 0 denotes expanding window



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
y_all = y_all(1:date_end_index);
y_date = y_date(1:date_end_index);
T_total = length(y_all);
L_test = find(y_date == test_start);
L_pre_end = L_test ;
%% load y_predict in first stage
load result/first_stage;
L_first_stage = length(First_stage);
for t_fs = 1:L_first_stage
    x_date_origin(t_fs) = First_stage(t_fs).date;
    x_all_origin(t_fs,:) = First_stage(t_fs).y_predict;
    R2_fs_origin(t_fs,:) = First_stage(t_fs).R2;
    nan_index =  find(isnan(x_all_origin(t_fs,:)));
    x_all_origin(t_fs,nan_index) = First_stage(t_fs).y_mean;
end

x_date_start_index = find(date_start == x_date_origin);
x_date = x_date_origin(x_date_start_index:end);
x_all = x_all_origin(x_date_start_index:end,:);


MK = size(x_all,2);
beta = zeros(MK,1);
RSS = 0;
RSS2 = 0;
TSS = 0;
count_est = 1;
Result_fix = struct;
%% main algorithm
for t =  L_pre_end:T_total - 1
    tt = t-L_pre_end+1;
    y_date_t = y_date(t,:);
    FS_index = find(y_date_t == x_date);
    
    
    if estimate_window >0 % fixed window
        start_index = max(1,FS_index-estimate_window);
        y_fit_fs_t = x_all(start_index:FS_index-1,:);
    else
        start_index = 1; % expanding window
        y_fit_fs_t = x_all(1:FS_index-1,:);
    end

    y_mean = mean(y_all(1:t));
    
    a_index = find(~isnan(sum(x_all(start_index:FS_index,:),1)));
    y_fit_fs_t = y_fit_fs_t(:,a_index) ;
    L_t = size(y_fit_fs_t,1);
    y_t = y_all(t-L_t+1:t) ;
    
    y_predict_fs_t(tt,:) = x_all(FS_index,:); % prediction by first stage
    
    y_true = y_all(t+1);
    
     if count_est == 1 % exclude intercept by demeaning the return
         [beta_all,beta_CV] = shrinkage_CV(y_fit_fs_t,y_t,[0.1:0.05:1],20,5,0); % estimate CEnet by Cross-Validation        
        [~,sort_index] = sort(beta_CV,'descend'); % sort by MSE

        beta = mean(beta_all(:,sort_index(end-5)),2);
        count_est = 0;
     
    beta(beta~=0) = 1/(length(find(beta~=0))); % equalized the non-zero weights
     end
    state_beta_save(tt,:) = beta;
    
    y_predict_save(tt) = y_predict_fs_t(tt,a_index) * beta(a_index)  ;

    
    RSS = RSS + (y_true - y_predict_save(tt))^2;  
    TSS = TSS + (y_true - mean(y_all(1:t)))^2;
    R2(tt) = 1-RSS/TSS;  
    R2_2(tt) = 1-RSS2/TSS;
    
    
    Result_fix(tt).date = y_date_t;
    Result_fix(tt).R2 = R2(tt);
    Result_fix(tt).R2_2 = R2_2(tt);
    Result_fix(tt).y_predict = y_predict_save(tt);
    
    Result_fix(tt).y_to_t =y_all(1:t);
    
    Result_fix(tt).y_mean = y_mean;

    Result_fix(tt).y_true = y_true;
    Result_fix(tt).beta = beta;
    Result_fix(tt).x_predict = y_predict_fs_t;
    count_est =count_est +1;
    toc
    disp([num2str((tt/(T_total-L_test))*100),'%']);
end
save('result/result_CEnet_a','Result_fix')
