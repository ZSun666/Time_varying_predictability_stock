%% this code is to replicate the forecasts result by LASSO
%% model setup
date_start = [1927,01];
test_start = [1988,01];
date_end = [2019,12];
estimate_window =0; % length of estimate window, 0 denotes expanding window

L_pre = 120;
L_train = 120;
lag = 0;


test_start = datetime([test_start,1]);
%% data load
addpath('result'); addpath('function'); addpath('data');

data_raw = readtable("new_dataset_full.xlsx");
var_name = data_raw.Properties.VariableNames;
y_date = datetime(num2str(data_raw{:,1}),'InputFormat','yyyyMM');

y_all = data_raw{:,2}*100;
x_all = data_raw{:,3:end};

%% truncate the obs series with date

date_start = datetime([date_start,1]);
date_end = datetime([date_end,1]);
date_start_index = find(date_start == y_date);
date_end_index = find(date_end == y_date);
y_all = y_all(date_start_index:date_end_index);
y_date = y_date(date_start_index:date_end_index);
T_total = length(y_all);



%% take the lag of obs
lag_one_index = [3:24]-2; % fin variables
lag_two_index = [25:41]-2; % monthly econ variables
lag_four_index = [42:45]-2; % quarterly econ variables
lag_portfolio_index = [46:145]-2; % Anomaly portfolio


x_all = x_all(date_start_index:date_end_index,:);




% take lag
x_all = [x_all(4:end-1,lag_one_index),x_all(3:end-2,lag_two_index),x_all(1:end-4,lag_four_index),x_all(4:end-1,lag_portfolio_index)];
% if fill_by_zero == 1
%     x_all(isnan(x_all)) = 0;
% 
% end
%-------------------------------
%x_all = x_all(:,[1:11,19,21,23,28,30,34,39:42]);
%-------------------------------

y_all = y_all(5:end);

y_date = y_date(5:end);

L_test = find(y_date == test_start);
L_pre_end = L_test;

T_total = length(y_all);

M = size(y_all,2);
K = size(x_all,2);
MK = K*M;
numa = MK*(MK-1)/2;



Result_ml = struct;
beta_save = zeros(MK,1);
RSS = 0;
TSS = 0;
tic
%% main algorithm

for t = L_pre_end:T_total - 1
    if estimate_window > 0
        start_index = max(1,t-estimate_window+1);
    else
        start_index  = 1;
    end
    tt = t - L_pre_end+1;
    % truncate data_set into training sample
  
    Y_mean = mean(y_all(1:t));
    Y_train = y_all(start_index:t) - Y_mean;
    Y_train_date = y_date(start_index:t,:);
    X_train = x_all(start_index:t+1,:);
    
    for m = 1:MK
        index_nan = find(isnan(X_train(:,m)));
        X_train(index_nan,m) = mean(X_train(~isnan(X_train(:,m)),m));
    end
    
    X_norm = [[1:length(X_train)]',(X_train)];
 
    X_t = X_norm(1:end-1,:);
    X_forecast = X_norm(end,:);
    [Ttrain,~] = size(Y_train);
    
    % fit model by GBRT （update model in every Jan）
    month_index = month(Y_train_date(t));
    if month_index == 1
    tree =  templateTree('PredictorSelection','interaction-curvature','Surrogate','on','Reproducible',true,'MaxNumSplits',5); 
    gbrt = fitrensemble(X_t,Y_train,'Method','LSBoost', ...
    'NumLearningCycles',100,'Learners',tree,'LearnRate',0.5);
    end
    beta = [];
    T = length(Y_train);
    
    % forecast by historical average (in training sample)
    y_mean = Y_mean;
    
    % forecast by GBRY
    x_predict = [X_forecast(end,:)];
    y_predict = predict(gbrt,x_predict) +Y_mean;
    
    y_true = y_all(t+1);
    RSS = RSS + (y_true - y_predict)^2;  

    TSS = TSS + (y_true - mean(y_all(1:t)))^2;
    R2(tt) = 1-RSS/TSS;  
    
    beta_save = beta;
    Result_ml(tt).date = Y_train_date(end,:); % 
    Result_ml(tt).x_predict = x_predict;
    Result_ml(tt).y_predict = y_predict;
    Result_ml(tt).y_mean = y_mean;
    Result_ml(tt).y_to_t = y_all(1:t);
    Result_ml(tt).y_true = y_all(t+1);
    Result_ml(tt).beta = beta_save;

    toc
    [num2str((tt/(T_total-L_pre_end))*100),'%']
end


save('result/result_gbrt','Result_ml');
