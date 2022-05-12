%% this code is to replicate the forecasts result by Elastic Net (ENet)
%% model setup
date_start = [1927,01];
date_end = [2019,12];
estimate_window =0; % length of estimate window, 0 denotes expanding window

L_pre = 120;



%% data load
addpath('result'); addpath('function'); addpath('data');


data_raw = readtable("new_dataset_full.xlsx");
var_name = data_raw.Properties.VariableNames;
y_date = datetime(num2str(data_raw{:,1}),'InputFormat','yyyyMM');

x_all = data_raw{:,3:end};
y_all = data_raw{:,2}*100;

%% truncate the obs series with date
date_start = datetime([date_start,1]);
date_end = datetime([date_end,1]);
date_start_index = find(date_start == y_date);
date_end_index = find(date_end == y_date);
y_all = y_all(date_start_index:date_end_index);
y_date = y_date(date_start_index:date_end_index);
T_total = length(y_all);


%% take the lag of obs
lag_one_index = [3:24]-2;
lag_two_index = [25:41]-2;
lag_four_index = [42:45]-2;
lag_portfolio_index = [46:145]-2;


x_all = x_all(date_start_index:date_end_index,:);

% take lag
x_all = [x_all(4:end-1,lag_one_index),x_all(3:end-2,lag_two_index),x_all(1:end-4,lag_four_index),x_all(4:end-1,lag_portfolio_index)];
y_all = y_all(5:end);

y_date = y_date(5:end);


T_total = length(y_all);

M = size(y_all,2);
K = size(x_all,2);
MK = K*M;
numa = MK*(MK-1)/2;


Result_fix = struct;
beta_save = zeros(MK,1);
count_est = 1;
tic
%% main algorithm

for t = L_pre+1:T_total - 1
    if estimate_window > 0
        start_index = max(1,t-estimate_window+1);
    else
        start_index  = 1;
    end
    tt = t - L_pre;
    % truncate data_set into training sample

    Y_mean = mean(y_all(1:t));
    Y_train = y_all(start_index:t);
    Y_train_date = y_date(start_index:t,:);
     X_train = x_all(start_index:t+1,:);
  
    
    % fill nan by mean
    for m = 1:MK
        index_nan = find(isnan(X_train(:,m)));
        X_train(index_nan,m) = mean(X_train(~isnan(X_train(:,m)),m));
        index_nan = find(isnan(X_train(:,m)));
        X_train(index_nan,m) = 0;
    end
   
    X_norm = trans_norm(X_train);

    X_norm(isnan(X_norm)) = 0;

    avail_index = find(~isnan(sum(X_train,1)));
    
    X_t = [X_norm(1:end-1,avail_index)];
    X_forecast = [X_norm(end,avail_index)];
    [Ttrain,~] = size(Y_train);
    
    % fit by ENet
    if count_est == 1
         [beta_all,beta_aicc,const_all] = shrinkage_CV(X_t,Y_train,[0.1:0.1:2],20,4,1); % set hyperparameters by Cross-Validation
    
        [~,min_index] = min(beta_aicc);% sort by MSE
        beta = beta_all(:,min_index);
        const = const_all(:,min_index);
        count_est = 0;
    end
    T = length(Y_train);
    count_est = count_est+1;
    % forecast by historical average (in training sample)
    y_mean = Y_mean;
    
    % forecast by ENet
    x_predict = [X_forecast(end,:)];
    y_predict = [x_predict]*beta(1:end) +const ;

    beta_save(avail_index) = beta(1:end);
    Result_fix(tt).date = Y_train_date(end,:); % 
    Result_fix(tt).x_predict = x_predict;
    Result_fix(tt).y_predict = y_predict;
    Result_fix(tt).y_mean = y_mean;
    Result_fix(tt).y_to_t = y_all(1:t);
    Result_fix(tt).y_true = y_all(t+1);
    Result_fix(tt).beta = beta_save;

    toc
    [num2str((tt/(T_total-estimate_window-1))*100),'%']
end
Result_fix(1).estimate_window = estimate_window;


save('result/result_ENet','Result_fix');
