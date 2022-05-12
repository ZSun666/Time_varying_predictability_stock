%% setup
draw_figure = 0;
model_type = 2; % 1 is lasso/pca/cEnet; 2 is tvc-predict % 3 is machine learning model 4. individual predictors

date_start  = [1988,1]; %long sample
date_end = [2019,12];
% date_subperiod ={[1976,2019],[1988,2019],[2003,2019],[2012,2019]};
date_subperiod ={[1988,2019],[2003,2019],[2012,2019]};


capital_init = 1e4; % initial value
lambda = 3 ; % risk aversion
max_weight = 1.5;
min_weight = 0;
%% load results
addpath('result')
addpath('function')
clear beta
clear signal_index  
clear error_model
clear error_benchmark

% load PCA
    load result_pca
    Result_pca = Result_fix;
% load ENet
    load result_ENet
    Result_ENet = Result_fix;
% load CENet
    load result_CENet
    Result_CENet = Result_fix; 
% load RF
    load result_RF
    Result_RF = Result_ml;     

% load NN1
    load result_NN1
    Result_NN1 = Result_ml;    
% load NN2
    load result_NN2
    Result_NN2 = Result_ml;    
% load dsc
    load result_dsc
    Result_dsc = Result_DSC;    


%% date range
date_start = datetime([date_start,1]);
date_end = datetime([date_end,1]);
date_range = date_start:calmonths(1):date_end;

MK = size(Result_dsc(1).x_predict,2);



%% temp matrix
Return = zeros(11,3);
Sharpe = zeros(11,3);

%% historical mean
T_length_bench = length(Result_pca);

count = 0;
RSS = 0;
TSS = 0;

for t = 1:T_length_bench  
    date_temp = Result_pca(t).date;
    if find(date_temp == date_range)        
        y_to_t = Result_pca(t).y_to_t/100;
    %         start_index = max(1,length(y_to_t) - L_window+1);
       
        error_date_hist(count+1) = date_temp;%+calmonths(1);
        y_predict = mean(y_to_t); 
        
        stock_return = (Result_pca(t).y_true/100);
        %y_var = Result(t).y_predict_var/10000;
        
        %y_var = var(y_to_t(end-119:end));
        y_var = var(y_to_t(end-59:end));
 
        
        weight_hist(count+1) = (1/lambda)*(y_predict/y_var);
        if weight_hist(count+1) > max_weight
            weight_hist(count+1) = max_weight;
        elseif weight_hist(count+1) < min_weight
            weight_hist(count+1) = min_weight;
        end
        
        if count > 0
            capital_hist(count+1) = (capital_hist(count)*weight_hist(count+1))*(exp(stock_return)) + capital_hist(count)*(1-weight_hist(count+1));
        else
            capital_hist(count+1) = (capital_init*weight_hist(count+1))*(exp(stock_return)) + capital_init*(1-weight_hist(count+1));
        end
        cum_return_hist(count+1) = (capital_hist(count+1)-(capital_init))/(capital_init);
        
        
       
   
        count = count + 1;

    end  
end
% compute MSFE, pvalue,R2 during each subperiods
for ii = 1:length(date_subperiod)
    date_subrange = date_subperiod{ii};
    date_start_sub = datetime([date_subrange(1),1,1]);
    date_end_sub = datetime([date_subrange(2),11,1]);
    index_start = find(date_start_sub == error_date_hist);
    index_end = find(date_end_sub == error_date_hist);
    year_diff = calyears(between(date_start_sub,date_end_sub,'years'));
    
    return_subperiod(ii) = (capital_hist(index_end)-capital_hist(index_start))/capital_hist(index_start);
    clear month_return_temp
    for l_temp = 1:length(capital_hist(index_start:index_end))-1
        month_return_temp(l_temp) = (capital_hist(index_start+l_temp) - capital_hist(index_start+l_temp-1))/capital_hist(index_start+l_temp-1);
    end
    sd_subperiod_temp = std(month_return_temp);
    Return(1,ii) = (1+return_subperiod(ii))^(1/year_diff)-1;
    Sharpe(1,ii) = Return(1,ii)/(sd_subperiod_temp*sqrt(year_diff));
end
%% evaluate pca    
T_length_pca = length(Result_pca);

count = 0;
RSS = 0;
TSS = 0;

for t = 1:T_length_pca  
    date_temp = Result_pca(t).date;
    if find(date_temp == date_range)        
        y_to_t = Result_pca(t).y_to_t/100;
    %         start_index = max(1,length(y_to_t) - L_window+1);
       
        error_date_pca(count+1) = date_temp;%+calmonths(1);
        y_predict = Result_pca(t).y_predict/100; 
        
        stock_return = (Result_pca(t).y_true/100);
        %y_var = Result(t).y_predict_var/10000;
        
        %y_var = var(y_to_t(end-119:end));
        y_var = var(y_to_t(end-59:end));
        y_mean = Result_pca(t).y_mean;
        
        weight_pca(count+1) = (1/lambda)*(y_predict/y_var);
        if weight_pca(count+1) > max_weight
            weight_pca(count+1) = max_weight;
        elseif weight_pca(count+1) < min_weight
            weight_pca(count+1) = min_weight;
        end
        
        if count > 0
            capital_pca(count+1) = (capital_pca(count)*weight_pca(count+1))*(exp(stock_return)) + capital_pca(count)*(1-weight_pca(count+1));
        else
            capital_pca(count+1) = (capital_init*weight_pca(count+1))*(exp(stock_return)) + capital_init*(1-weight_pca(count+1));
        end
        cum_return_pca(count+1) = (capital_pca(count+1)-(capital_init))/(capital_init);
        
        
       
   
        count = count + 1;

    end  
end
% compute MSFE, pvalue,R2 during each subperiods
for ii = 1:length(date_subperiod)
    date_subrange = date_subperiod{ii};
    date_start_sub = datetime([date_subrange(1),1,1]);
    date_end_sub = datetime([date_subrange(2),11,1]);
    index_start = find(date_start_sub == error_date_pca);
    index_end = find(date_end_sub == error_date_pca);
    year_diff = calyears(between(date_start_sub,date_end_sub,'years'));
    
    return_subperiod(ii) = (capital_pca(index_end)-capital_pca(index_start))/capital_pca(index_start);
    clear month_return_temp
    for l_temp = 1:length(capital_pca(index_start:index_end))-1
        month_return_temp(l_temp) = (capital_pca(index_start+l_temp) - capital_pca(index_start+l_temp-1))/capital_pca(index_start+l_temp-1);
    end
    sd_subperiod_temp = std(month_return_temp);
    Return(2,ii) = (1+return_subperiod(ii))^(1/year_diff)-1;
    Sharpe(2,ii) = Return(2,ii)/(sd_subperiod_temp*sqrt(year_diff));
end
%% evaluate ENet
T_length_ENet = length(Result_ENet);

count = 0;
RSS = 0;
TSS = 0;

for t = 1:T_length_ENet
    % datetime    
    date_temp = Result_ENet(t).date;
    if find(date_temp == date_range)        
        y_to_t = Result_ENet(t).y_to_t/100;
    %         start_index = max(1,length(y_to_t) - L_window+1);
       
        error_date_ENet(count+1) = date_temp;%+calmonths(1);
        y_predict = Result_ENet(t).y_predict/100; 
        
        stock_return = (Result_ENet(t).y_true/100);
        %y_var = Result(t).y_predict_var/10000;
        
        %y_var = var(y_to_t(end-119:end));
        y_var = var(y_to_t(end-59:end));
        y_mean = Result_ENet(t).y_mean;
        
        weight_ENet(count+1) = (1/lambda)*(y_predict/y_var);
        if weight_ENet(count+1) > max_weight
            weight_ENet(count+1) = max_weight;
        elseif weight_ENet(count+1) < min_weight
            weight_ENet(count+1) = min_weight;
        end
        
        if count > 0
            capital_ENet(count+1) = (capital_ENet(count)*weight_ENet(count+1))*(exp(stock_return)) + capital_ENet(count)*(1-weight_ENet(count+1));
        else
            capital_ENet(count+1) = (capital_init*weight_ENet(count+1))*(exp(stock_return))+  capital_init*(1-weight_ENet(count+1));
        end
        cum_return_ENet(count+1) = (capital_ENet(count+1)-(capital_init))/(capital_init);
        
        
       
   
        count = count + 1;

    end  
end
% compute MSFE, pvalue,R2 during each subperiods
for ii = 1:length(date_subperiod)
    date_subrange = date_subperiod{ii};
    date_start_sub = datetime([date_subrange(1),1,1]);
    date_end_sub = datetime([date_subrange(2),11,1]);
    index_start = find(date_start_sub == error_date_ENet);
    index_end = find(date_end_sub == error_date_ENet);
    year_diff = calyears(between(date_start_sub,date_end_sub,'years'));
    
    return_subperiod(ii) = (capital_ENet(index_end)-capital_ENet(index_start))/capital_ENet(index_start);
    clear month_return_temp
    for l_temp = 1:length(capital_ENet(index_start:index_end))-1
        month_return_temp(l_temp) = (capital_ENet(index_start+l_temp) - capital_ENet(index_start+l_temp-1))/capital_ENet(index_start+l_temp-1);
    end
    sd_subperiod_temp = std(month_return_temp);
    Return(3,ii) = (1+return_subperiod(ii))^(1/year_diff)-1;
    Sharpe(3,ii) = Return(3,ii)/(sd_subperiod_temp*sqrt(year_diff));
end
%% evaluate CENet
T_length_CENet = length(Result_CENet);

count = 0;
RSS = 0;
TSS = 0;

for t = 1:T_length_CENet
    % datetime    
    date_temp = Result_CENet(t).date;
    if find(date_temp == date_range)        
        y_to_t = Result_CENet(t).y_to_t/100;
    %         start_index = max(1,length(y_to_t) - L_window+1);
       
        error_date_CENet(count+1) = date_temp;%+calmonths(1);
        y_predict = Result_CENet(t).y_predict/100; 
        
        stock_return = (Result_CENet(t).y_true/100);
        %y_var = Result(t).y_predict_var/10000;
        
        %y_var = var(y_to_t(end-119:end));
        y_var = var(y_to_t(end-59:end));
        y_mean = Result_CENet(t).y_mean;
        
        weight_CENet(count+1) = (1/lambda)*(y_predict/y_var);
        if weight_CENet(count+1) > max_weight
            weight_CENet(count+1) = max_weight;
        elseif weight_CENet(count+1) < min_weight
            weight_CENet(count+1) = min_weight;
        end
        
        if count > 0
            capital_CENet(count+1) = (capital_CENet(count)*weight_CENet(count+1))*(exp(stock_return)) + capital_CENet(count)*(1-weight_CENet(count+1));
        else
            capital_CENet(count+1) = (capital_init*weight_CENet(count+1))*(exp(stock_return)) +  capital_init*(1-weight_CENet(count+1));
        end
        cum_return_CENet(count+1) = (capital_CENet(count+1)-(capital_init))/(capital_init);
        
        
       
   
        count = count + 1;

    end  
end
% compute MSFE, pvalue,R2 during each subperiods
for ii = 1:length(date_subperiod)
    date_subrange = date_subperiod{ii};
    date_start_sub = datetime([date_subrange(1),1,1]);
    date_end_sub = datetime([date_subrange(2),11,1]);
    index_start = find(date_start_sub == error_date_CENet);
    index_end = find(date_end_sub == error_date_CENet);
    year_diff = calyears(between(date_start_sub,date_end_sub,'years'));
    
    return_subperiod(ii) = (capital_CENet(index_end)-capital_CENet(index_start))/capital_CENet(index_start);
    clear month_return_temp
    for l_temp = 1:length(capital_CENet(index_start:index_end))-1
        month_return_temp(l_temp) = (capital_CENet(index_start+l_temp) - capital_CENet(index_start+l_temp-1))/capital_CENet(index_start+l_temp-1);
    end
    sd_subperiod_temp = std(month_return_temp);
    Return(4,ii) = (1+return_subperiod(ii))^(1/year_diff)-1;
    Sharpe(4,ii) = Return(4,ii)/(sd_subperiod_temp*sqrt(year_diff));
end



%% evaluate RF
T_length_RF = length(Result_RF);

count = 0;
RSS = 0;
TSS = 0;

for t = 1:T_length_RF
    % datetime    
    date_temp = Result_RF(t).date;
    if find(date_temp == date_range)        
        y_to_t = Result_RF(t).y_to_t/100;
    %         start_index = max(1,length(y_to_t) - L_window+1);
       
        error_date_RF(count+1) = date_temp;%+calmonths(1);
        y_predict = Result_RF(t).y_predict/100; 
        
        stock_return = (Result_RF(t).y_true/100);
        %y_var = Result(t).y_predict_var/10000;
        
        %y_var = var(y_to_t(end-119:end));
        y_var = var(y_to_t(end-59:end));
        y_mean = Result_RF(t).y_mean;
        
        weight_RF(count+1) = (1/lambda)*(y_predict/y_var);
        if weight_RF(count+1) > max_weight
            weight_RF(count+1) = max_weight;
        elseif weight_RF(count+1) < min_weight
            weight_RF(count+1) = min_weight;
        end
        
        if count > 0
            capital_RF(count+1) = (capital_RF(count)*weight_RF(count+1))*(exp(stock_return)) + capital_RF(count)*(1-weight_RF(count+1));
        else
            capital_RF(count+1) = (capital_init*weight_RF(count+1))*(exp(stock_return)) +  capital_init*(1-weight_RF(count+1));
        end
        cum_return_RF(count+1) = (capital_RF(count+1)-(capital_init))/(capital_init);
        
        
       
   
        count = count + 1;

    end  
end
% compute MSFE, pvalue,R2 during each subperiods
for ii = 1:length(date_subperiod)
    date_subrange = date_subperiod{ii};
    date_start_sub = datetime([date_subrange(1),1,1]);
    date_end_sub = datetime([date_subrange(2),11,1]);
    index_start = find(date_start_sub == error_date_RF);
    index_end = find(date_end_sub == error_date_RF);
    year_diff = calyears(between(date_start_sub,date_end_sub,'years'));
    
    return_subperiod(ii) = (capital_RF(index_end)-capital_RF(index_start))/capital_RF(index_start);
    clear month_return_temp
    for l_temp = 1:length(capital_RF(index_start:index_end))-1
        month_return_temp(l_temp) = (capital_RF(index_start+l_temp) - capital_RF(index_start+l_temp-1))/capital_RF(index_start+l_temp-1);
    end
    sd_subperiod_temp = std(month_return_temp);
    Return(5,ii) = (1+return_subperiod(ii))^(1/year_diff)-1;
    Sharpe(5,ii) = Return(5,ii)/(sd_subperiod_temp*sqrt(year_diff));
end
%% evaluate NN1
T_length_NN1 = length(Result_NN1);

count = 0;
RSS = 0;
TSS = 0;

for t = 1:T_length_NN1
    % datetime    
    date_temp = Result_NN1(t).date;
    if find(date_temp == date_range)        
        y_to_t = Result_NN1(t).y_to_t/100;
    %         start_index = max(1,length(y_to_t) - L_window+1);
       
        error_date_NN1(count+1) = date_temp;%+calmonths(1);
        y_predict = Result_NN1(t).y_predict/100; 
        
        stock_return = (Result_NN1(t).y_true/100);
        %y_var = Result(t).y_predict_var/10000;
        
        %y_var = var(y_to_t(end-119:end));
        y_var = var(y_to_t(end-59:end));
        y_mean = Result_NN1(t).y_mean;
        
        weight_NN1(count+1) = (1/lambda)*(y_predict/y_var);
        if weight_NN1(count+1) > max_weight
            weight_NN1(count+1) = max_weight;
        elseif weight_NN1(count+1) < min_weight
            weight_NN1(count+1) = min_weight;
        end
        
        if count > 0
            capital_NN1(count+1) = (capital_NN1(count)*weight_NN1(count+1))*(exp(stock_return)) + capital_NN1(count)*(1-weight_NN1(count+1));
        else
            capital_NN1(count+1) = (capital_init*weight_NN1(count+1))*(exp(stock_return))+ capital_init*(1-weight_NN1(count+1));
        end
        cum_return_NN1(count+1) = (capital_NN1(count+1)-(capital_init))/(capital_init);
        
        
       
   
        count = count + 1;

    end  
end
% compute MSFE, pvalue,R2 during each subperiods
for ii = 1:length(date_subperiod)
    date_subrange = date_subperiod{ii};
    date_start_sub = datetime([date_subrange(1),1,1]);
    date_end_sub = datetime([date_subrange(2),11,1]);
    index_start = find(date_start_sub == error_date_NN1);
    index_end = find(date_end_sub == error_date_NN1);
    year_diff = calyears(between(date_start_sub,date_end_sub,'years'));
    
    return_subperiod(ii) = (capital_NN1(index_end)-capital_NN1(index_start))/capital_NN1(index_start);
    clear month_return_temp
    for l_temp = 1:length(capital_NN1(index_start:index_end))-1
        month_return_temp(l_temp) = (capital_NN1(index_start+l_temp) - capital_NN1(index_start+l_temp-1))/capital_NN1(index_start+l_temp-1);
    end
    sd_subperiod_temp = std(month_return_temp);
    Return(6,ii) = (1+return_subperiod(ii))^(1/year_diff)-1;
    Sharpe(6,ii) = Return(6,ii)/(sd_subperiod_temp*sqrt(year_diff));
end
%% evaluate NN2
T_length_NN2 = length(Result_NN2);

count = 0;
RSS = 0;
TSS = 0;

for t = 1:T_length_NN2
    % datetime    
    date_temp = Result_NN2(t).date;
    if find(date_temp == date_range)        
        y_to_t = Result_NN2(t).y_to_t/100;
    %         start_index = max(1,length(y_to_t) - L_window+1);
       
        error_date_NN2(count+1) = date_temp;%+calmonths(1);
        y_predict = Result_NN2(t).y_predict/100; 
        
        stock_return = (Result_NN2(t).y_true/100);
        
        y_var = var(y_to_t(end-59:end));
        y_mean = Result_NN2(t).y_mean;
        
        weight_NN2(count+1) = (1/lambda)*(y_predict/y_var);
        if weight_NN2(count+1) > max_weight
            weight_NN2(count+1) = max_weight;
        elseif weight_NN2(count+1) < min_weight
            weight_NN2(count+1) = min_weight;
        end
        
        if count > 0
            capital_NN2(count+1) = (capital_NN2(count)*weight_NN2(count+1))*(exp(stock_return)) + capital_NN2(count)*(1-weight_NN2(count+1));
        else
            capital_NN2(count+1) = (capital_init*weight_NN2(count+1))*(exp(stock_return)) + capital_init*(1-weight_NN2(count+1));
        end
        cum_return_NN2(count+1) = (capital_NN2(count+1)-(capital_init))/(capital_init);
        
        
       
   
        count = count + 1;

    end  
end
% compute MSFE, pvalue,R2 during each subperiods
for ii = 1:length(date_subperiod)
    date_subrange = date_subperiod{ii};
    date_start_sub = datetime([date_subrange(1),1,1]);
    date_end_sub = datetime([date_subrange(2),11,1]);
    index_start = find(date_start_sub == error_date_NN2);
    index_end = find(date_end_sub == error_date_NN2);
    year_diff = calyears(between(date_start_sub,date_end_sub,'years'));
    
    return_subperiod(ii) = (capital_NN2(index_end)-capital_NN2(index_start))/capital_NN2(index_start);
    clear month_return_temp
    for l_temp = 1:length(capital_NN2(index_start:index_end))-1
        month_return_temp(l_temp) = (capital_NN2(index_start+l_temp) - capital_NN2(index_start+l_temp-1))/capital_NN2(index_start+l_temp-1);
    end
    sd_subperiod_temp = std(month_return_temp);
    Return(7,ii) = (1+return_subperiod(ii))^(1/year_diff)-1;
    Sharpe(7,ii) = Return(7,ii)/(sd_subperiod_temp*sqrt(year_diff));
end
%% evaluate DSC
T_length_DSC = length(Result_dsc);

count = 0;
RSS = 0;
TSS = 0;

for t = 1:T_length_DSC
    % datetime    
    date_temp = Result_dsc(t).date;

    if find(date_temp == date_range)        

        y_predict_DSC_ORIG = Result_dsc(t).y_predict/100;
        y_predict_DSC_NORM = Result_dsc(t).y_predict_norm/100;
        y_predict_DSC_EQ =   Result_dsc(t).y_predict_equal/100;
        
        y_to_t = Result_dsc(t).y_to_t/100;

        stock_return = (Result_dsc(t).y_true/100);

        error_date_DSC(count+1) = date_temp;%+calmonths(1);
        
        y_var = var(y_to_t(end-59:end));
        y_mean = Result_dsc(t).y_mean;
        
        weight_DSC_ORIG(count+1) = (1/lambda)*(y_predict_DSC_ORIG/y_var);
        weight_DSC_NORM(count+1) = (1/lambda)*(y_predict_DSC_NORM/y_var);
        weight_DSC_EQ(count+1) = (1/lambda)*(y_predict_DSC_EQ/y_var);
        if weight_DSC_ORIG(count+1) > max_weight
            weight_DSC_ORIG(count+1) = max_weight;
        elseif weight_DSC_ORIG(count+1) < min_weight
            weight_DSC_ORIG(count+1) = min_weight;
        end
        if weight_DSC_NORM(count+1) > max_weight
            weight_DSC_NORM(count+1) = max_weight;
        elseif weight_DSC_NORM(count+1) < min_weight
            weight_DSC_NORM(count+1) = min_weight;
        end
        if weight_DSC_EQ(count+1) > max_weight
            weight_DSC_EQ(count+1) = max_weight;
        elseif weight_DSC_EQ(count+1) < min_weight
            weight_DSC_EQ(count+1) = min_weight;
        end
        
        if count > 0
            capital_DSC_ORIG(count+1) = (capital_DSC_ORIG(count)*weight_DSC_ORIG(count+1))*(exp(stock_return)) + capital_DSC_ORIG(count)*(1-weight_DSC_ORIG(count+1));
            capital_DSC_NORM(count+1) = (capital_DSC_NORM(count)*weight_DSC_NORM(count+1))*(exp(stock_return)) + capital_DSC_NORM(count)*(1-weight_DSC_NORM(count+1));
            capital_DSC_EQ(count+1) = (capital_DSC_EQ(count)*weight_DSC_EQ(count+1))*(exp(stock_return)) + capital_DSC_EQ(count)*(1-weight_DSC_EQ(count+1));
        else
            capital_DSC_ORIG(count+1) = (capital_init*weight_DSC_ORIG(count+1))*(exp(stock_return)) + capital_init*(1-weight_DSC_ORIG(count+1));
            capital_DSC_NORM(count+1) = (capital_init*weight_DSC_NORM(count+1))*(exp(stock_return))+ capital_init*(1-weight_DSC_NORM(count+1));
            capital_DSC_EQ(count+1) = (capital_init*weight_DSC_EQ(count+1))*(exp(stock_return))+ capital_init*(1-weight_DSC_EQ(count+1));
        end
        cum_return_DSC_ORIG(count+1) = (capital_DSC_ORIG(count+1)-(capital_init))/(capital_init);
        cum_return_DSC_NORM(count+1) = (capital_DSC_NORM(count+1)-(capital_init))/(capital_init);
        cum_return_DSC_EQ(count+1) = (capital_DSC_EQ(count+1)-(capital_init))/(capital_init);
   
        count = count + 1;

    end  
end

% compute MSFE, pvalue,R2 during each subperiods
for ii = 1:length(date_subperiod)
    date_subrange = date_subperiod{ii};
    date_start_sub = datetime([date_subrange(1),1,1]);
    date_end_sub = datetime([date_subrange(2),11,1]);
    index_start = find(date_start_sub == error_date_DSC);
    index_end = find(date_end_sub == error_date_DSC);
    year_diff = calyears(between(date_start_sub,date_end_sub,'years'));
    
    % DSC_ORIG
    return_subperiod(ii) = (capital_DSC_ORIG(index_end)-capital_DSC_ORIG(index_start))/capital_DSC_ORIG(index_start);
    clear month_return_temp
    for l_temp = 1:length(capital_DSC_ORIG(index_start:index_end))-1
        month_return_temp(l_temp) = (capital_DSC_ORIG(index_start+l_temp) - capital_DSC_ORIG(index_start+l_temp-1))/capital_DSC_ORIG(index_start+l_temp-1);
    end
    sd_subperiod_temp = std(month_return_temp);
    Return(8,ii) = (1+return_subperiod(ii))^(1/year_diff)-1;
    Sharpe(8,ii) = Return(8,ii)/(sd_subperiod_temp*sqrt(year_diff));


    % DSC_NORM
    return_subperiod(ii) = (capital_DSC_NORM(index_end)-capital_DSC_NORM(index_start))/capital_DSC_NORM(index_start);
    clear month_return_temp
    for l_temp = 1:length(capital_DSC_NORM(index_start:index_end))-1
        month_return_temp(l_temp) = (capital_DSC_NORM(index_start+l_temp) - capital_DSC_NORM(index_start+l_temp-1))/capital_DSC_NORM(index_start+l_temp-1);
    end
    sd_subperiod_temp = std(month_return_temp);
    Return(9,ii) = (1+return_subperiod(ii))^(1/year_diff)-1;
    Sharpe(9,ii) = Return(9,ii)/(sd_subperiod_temp*sqrt(year_diff));

    % DSC_EQ
    return_subperiod(ii) = (capital_DSC_EQ(index_end)-capital_DSC_EQ(index_start))/capital_DSC_EQ(index_start);
    clear month_return_temp
    for l_temp = 1:length(capital_DSC_EQ(index_start:index_end))-1
        month_return_temp(l_temp) = (capital_DSC_EQ(index_start+l_temp) - capital_DSC_EQ(index_start+l_temp-1))/capital_DSC_EQ(index_start+l_temp-1);
    end
    sd_subperiod_temp = std(month_return_temp);
    Return(10,ii) = (1+return_subperiod(ii))^(1/year_diff)-1;
    Sharpe(10,ii) = Return(10,ii)/(sd_subperiod_temp*sqrt(year_diff));

end
%% statististical Evaluation
econ_output = table;
model_name = {'Hist','PCA','ENet','CENet','RF','NN1','NN2','DSC-ORIG','DSC-NORM','DSC-EQ'};
for i_model = 1:10
    econ_output.model_name{i_model}= model_name{i_model};
    % subperiod 1988-2019
    econ_output.Return_1{i_model}= Return(i_model,1);
    econ_output.Sharpe_1{i_model}= Sharpe(i_model,1);
    % subperiod 2003-2019
    econ_output.Return_2{i_model}= Return(i_model,2);
    econ_output.Sharpe_2{i_model}= Sharpe(i_model,2);
    % subperiod 2012-2019
    econ_output.Return_3{i_model}= Return(i_model,3);
    econ_output.Sharpe_3{i_model}= Sharpe(i_model,3);
end

%% draw figure Cumulative return of Portfolio

    figure('Position',[10,10,1500,900],'InvertHardcopy','off','Color',[1 1 1]);
    n_d = count;
    x(1) = date_start;
    for i = 2:n_d
        x(i) = x(i-1)+calmonths(1);
    end
    y_bench = cum_return_hist;
    
    y_0 = cum_return_DSC_NORM;
    
    y_1 = cum_return_pca;
    y_2 = cum_return_ENet;
    y_3 = cum_return_CENet;
    y_4 = cum_return_RF;
    y_5 = cum_return_NN1;
    
    
    
    hold on
    box on
    plot(x,y_bench,'LineWidth',2);  
    plot(x,y_0,'LineWidth',2);
    plot(x,y_1,'LineWidth',2);
    plot(x,y_2,'LineWidth',2);
    plot(x,y_3,'LineWidth',2);
    plot(x,y_4,'LineWidth',2);
    plot(x,y_5,'LineWidth',2);
    
    legend('Historic mean','DSC','PCA','ENet','CENet','RF','NN','AutoUpdate','off','Location','northeast','Fontweight','bold')
%     plot(x,y_bench,'LineWidth',2,'Color','black','LineStyle','- -');
    
    xlim([x(1) x(end)]);   
    datetick('x','mmm-yy','keeplimits','keepticks')
    
    % NBER business cycle
    yl = ylim;
    x_1 = [x(31),x(39),x(39),x(31)];
    % first shaded area
    y_0 = [yl(1),yl(1),yl(2),yl(2)];
    p1 = patch(x_1,y_0,[0.501960784313725 0.501960784313725 0.501960784313725],'EdgeColor','none', 'FaceAlpha',0.5);
    
    x_2 = [x(159),x(167),x(167),x(159)];
    p2 = patch(x_2,y_0,[0.501960784313725 0.501960784313725 0.501960784313725],'EdgeColor','none', 'FaceAlpha',0.5);
    
    x_3 = [x(240),x(258),x(258),x(240)];
    p3 = patch(x_3,y_0,[0.501960784313725 0.501960784313725 0.501960784313725],'EdgeColor','none', 'FaceAlpha',0.5);
%     x_6 = [x(530),x(end),x(end),x(530)];
%     p6 = patch(x_6,y_0,[0.501960784313725 0.501960784313725 0.501960784313725],'EdgeColor','none', 'FaceAlpha',0.5);