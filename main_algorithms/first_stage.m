tic
% this version use different normalized way  ----- by pre-sample mean and
% var
%% model setup
date_start = [1927,01];
test_start = [1947,01];
% date_end = [2020,12];
date_end = [2019,12];

estimate_window = 0; % 0 is expanding window
L_train = 120;
%% data load
addpath('result'); addpath('function'); addpath('data');
data_raw = readtable("new_dataset_full.xlsx");


var_name = data_raw.Properties.VariableNames;
y_date = datetime(num2str(data_raw{:,1}),'InputFormat','yyyyMM');

y_all = data_raw{:,2}*100;
x_all = data_raw{:,3:end};

%% truncate the obs series with date
date_start = datetime([date_start,1]);
test_start = datetime([test_start,1]);
date_end = datetime([date_end,1]);
date_start_index = find(date_start == y_date);
test_start_index = find(test_start == y_date);


date_end_index = find(date_end == y_date);

y_date = y_date(date_start_index:date_end_index);
y_all = y_all(date_start_index:date_end_index);
x_all = x_all(date_start_index:date_end_index,:);
%% take the lag
lag_one_index = [3:24]-2; % fin variables
lag_two_index = [25:41]-2; % monthly econ variables
lag_four_index = [42:45]-2; % quarterly econ variables
% lag_one_portfolio = [46:145]-2; % Anomaly portfolio
lag_one_portfolio = [46:145]-2; % Anomaly portfolio


x_all = [x_all(4:end-1,lag_one_index),x_all(3:end-2,lag_two_index),x_all(1:end-4,lag_four_index),x_all(4:end-1,lag_one_portfolio)];
y_all = y_all(5:end);

y_date = y_date(5:end);


T_total = length(y_all);

M = size(y_all,2);
K = size(x_all,2);%+size(return_all,2);
MK = K*M;
numa = MK*(MK-1)/2;

%% storage matrix

First_stage = struct;
RSS = zeros(MK,1);
TSS = zeros(MK,1);

%% pre-sample estimation
L_test = find(y_date == test_start);



%% main algorithm
for t = L_test:T_total-1
    
    tt = t - L_test+1;
    % obtain all information available at t
    start_index = 1;
    y_mean = mean(y_all(1:t));
    
    if estimate_window == 0
        start_index = 1;
    else
        start_index = max(1,t-estimate_window-1);
    end
    x_all_t = [x_all(start_index:t+1,:)];
    y_predict = nan(MK,1);
    x_predict = x_all_t(end,:); 
    x_train = x_all_t(1:end-1,:);
    
    y_train = y_all(start_index:t)-mean(y_all(1:t)); % demean return
    y_train_date = y_date(t,:);
    
    y_true = y_all(t+1,:);
    % estimate the predictive regression with each predictor
    for m = 1:MK
        index_obs = find(~isnan(x_train(:,m)));
        if length(index_obs)>10
            X_m = [x_train(index_obs,m)];
            y_train_m = y_train(index_obs);
            beta_temp = (X_m'*X_m)\(X_m'*y_train_m);
 
            y_predict(m) = x_predict(m) * beta_temp+y_mean;

            if ~isnan(y_predict(m))
                RSS(m) = RSS(m) + (y_true - (y_predict(m)))^2;
                TSS(m) = TSS(m) + (y_true - mean(y_all(1:t)))^2;
            elseif isnan(y_predict(m)) && RSS(m) ~= 0
                RSS(m) = nan;
            end
        end
    end
   
    First_stage(tt).R2 = ones(MK,1)-RSS./TSS;  
  
   
    First_stage(tt).date = y_train_date(end,:); % 
    First_stage(tt).x_predict = x_predict;
    First_stage(tt).y_predict = y_predict;
   
    First_stage(tt).y_to_t =y_all(1:t);
  
    First_stage(tt).y_mean = y_mean;
   
    First_stage(tt).y_true = y_true;

    toc
    disp([num2str((tt/(T_total-L_test))*100),'%']);
end

save('result/first_stage','First_stage');

