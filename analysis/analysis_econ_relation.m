%% setup
draw_figure = 0;
model_type = 2; % 1 is lasso/pca/cEnet; 2 is tvc-predict % 3 is machine learning model 4. individual predictors

date_start  = [1988,1]; %long sample
date_end = [2019,12];

date_subperiod ={[1988,2019],[2003,2019],[2012,2019]};
L_window = 240; % fixed window to compute y_mean and y_var;

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
% load GBRT
    load result_GBRT
    Result_GBRT = Result_ml;    
% load NN1
    load result_NN1
    Result_NN1 = Result_ml;    
% load NN2
    load result_NN2
    Result_NN2 = Result_ml;    
% load dsc
    load result_DSC
    Result_dsc = Result_DSC;    


%% date range
date_start = datetime([date_start,1]);
date_end = datetime([date_end,1]);
date_range = date_start:calmonths(1):date_end;

MK = size(Result_dsc(1).x_predict,2);



%% temp matrix
MSFE = zeros(4,3);
pvalue = zeros(4,3);
R2 = zeros(4,3);

%% evaluate CENet
T_length_CENet = length(Result_CENet);

count = 0;
RSS = 0;
TSS = 0;

for t = 1:T_length_CENet
    % datetime    
    date_temp = Result_CENet(t).date;

    if find(date_temp == date_range)        
        beta_CENet(count+1,:) = Result_CENet(t).beta;

 
        y_predict_CENet(count+1,:) = Result_CENet(t).y_predict;
        y_true_CENet(count+1) = Result_CENet(t).y_true(end);
        y_to_t = Result_CENet(t).y_to_t;
        y_mean = mean(y_to_t(1:end));

        error_date_CENet(count+1) = date_temp;%+calmonths(1);
        error_CENet(count+1) = (y_predict_CENet(count+1)-y_true_CENet(count+1))^2;
   
        error_benchmark_CENet(count+1) = (y_mean - y_true_CENet(count+1))^2;
        RSS = RSS + error_CENet(count+1);
        TSS = TSS + error_benchmark_CENet(count+1);
        Diff_CSE_CENet(count+1) = (RSS) - (TSS);
        R_squared_CENet(count+1) = 1- RSS/TSS;
   
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
    RSS_temp = sum(error_CENet(index_start:index_end));
    TSS_temp = sum(error_benchmark_CENet(index_start:index_end));
   
    R2(1,ii) = 1-RSS_temp./TSS_temp;
    MSFE(1,ii) = sqrt(mean(error_CENet(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_CENet(index_start:index_end))',sqrt(error_benchmark_CENet(index_start:index_end))',1);
    pvalue(1,ii) = 1-mvncdf(abs(DM_test_temp));
end



%% evaluate DSC
T_length_DSC = length(Result_dsc);

count = 0;
RSS_ORIG = 0;
RSS_NORM = 0;
RSS_EQ = 0;
TSS = 0;

for t = 1:T_length_DSC
    % datetime    
    date_temp = Result_dsc(t).date;

    if find(date_temp == date_range)        

        y_predict_DSC_ORIG(count+1,:) = Result_dsc(t).y_predict;
        y_predict_DSC_NORM(count+1,:) = Result_dsc(t).y_predict_norm;
        y_predict_DSC_EQ(count+1,:) =   Result_dsc(t).y_predict_equal;
        y_true_DSC(count+1) = Result_dsc(t).y_true(end);
        y_to_t = Result_dsc(t).y_to_t;
        y_mean = mean(y_to_t(1:end));

        error_date_DSC(count+1) = date_temp;%+calmonths(1);
        error_DSC_ORIG(count+1) = (y_predict_DSC_ORIG(count+1)-y_true_DSC(count+1))^2;
        error_DSC_NORM(count+1) = (y_predict_DSC_NORM(count+1)-y_true_DSC(count+1))^2;
        error_DSC_EQ(count+1) = (y_predict_DSC_EQ(count+1)-y_true_DSC(count+1))^2;

        error_benchmark_DSC(count+1) = (y_mean - y_true_DSC(count+1))^2;

        RSS_ORIG = RSS_ORIG + error_DSC_ORIG(count+1);
        RSS_NORM = RSS_NORM + error_DSC_NORM(count+1);
        RSS_EQ = RSS_EQ + error_DSC_EQ(count+1);
        TSS = TSS + error_benchmark_DSC(count+1);
        Diff_CSE_DSC(count+1) = (RSS_NORM) - (TSS);

        R_squared_DSC_ORIG(count+1) = 1- RSS_ORIG/TSS;
        R_squared_DSC_NORM(count+1) = 1- RSS_NORM/TSS;
        R_squared_DSC_EQ(count+1) = 1- RSS_EQ/TSS;

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
    TSS_temp = sum(error_benchmark_DSC(index_start:index_end));
    
    % DSC_ORIG
    RSS_temp = sum(error_DSC_ORIG(index_start:index_end));
    R2(2,ii) = 1-RSS_temp./TSS_temp;
    MSFE(2,ii) = sqrt(mean(error_DSC_ORIG(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_DSC_ORIG(index_start:index_end))',sqrt(error_benchmark_DSC(index_start:index_end))',1);
    pvalue(2,ii) = 1-mvncdf(abs(DM_test_temp));

    % DSC_NORM
    RSS_temp = sum(error_DSC_NORM(index_start:index_end));
    R2(3,ii) = 1-RSS_temp./TSS_temp;
    MSFE(3,ii) = sqrt(mean(error_DSC_NORM(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_DSC_NORM(index_start:index_end))',sqrt(error_benchmark_DSC(index_start:index_end))',1);
    pvalue(3,ii) = 1-mvncdf(abs(DM_test_temp));

    % DSC_EQ
    RSS_temp = sum(error_DSC_EQ(index_start:index_end));
    R2(4,ii) = 1-RSS_temp./TSS_temp;
    MSFE(4,ii) = sqrt(mean(error_DSC_EQ(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_DSC_EQ(index_start:index_end))',sqrt(error_benchmark_DSC(index_start:index_end))',1);
    pvalue(4,ii) = 1-mvncdf(abs(DM_test_temp));
end

%% load econ variable

econ_var = readtable('econ_variable.xlsx'); % load econ var

econ_date = econ_var{:,1};
length_econ = length(econ_date);
tt = 1;
for t= 1:length_econ
    econ_date_t = num2str(econ_date(t));
    econ_date_t = datetime([str2num(econ_date_t(1:4)),str2num(econ_date_t(5:6)),1]);
    econ_date_save(t) = econ_date_t;
    econ_var_num(t,:) = econ_var{t,[2,3,7]};
    date_temp = econ_date_save(t);

    if find(date_temp == date_range)
        econ_final(tt,:) = econ_var_num(t,:); % get the variable during the evaluation period
        tt = tt+1;
    end
end

%% Table: Link to the real economy

for k = 1:3
    [corr_temp,corr_temp_p] = corrcoef(econ_final(1:end-1,k),R_squared_CENet(1:end)');
    corr_econ(1,k) = corr_temp(2,1);
    corr_econ_p(1,k) = corr_temp_p(2,1);
    [corr_temp,corr_temp_p] = corrcoef(econ_final(1:end-1,k),R_squared_DSC_NORM(1:end)');
    corr_econ(2,k) = corr_temp(2,1);
    corr_econ_p(2,k) = corr_temp_p(2,1);
    [corr_temp,corr_temp_p] = corrcoef(econ_final(1:end-1,k),R_squared_DSC_EQ(1:end)');
    corr_econ(3,k) = corr_temp(2,1);
    corr_econ_p(3,k) = corr_temp_p(2,1);
end

%% Table: Out-of-sample R2 during different regimes

    % sort by ADS
    [~,sort_index] = sort(econ_final(1:end-1,1),'descend');
    part = floor(length(sort_index)/3);
    good_index = sort_index(1:part);
    SSE_temp = sum(error_CENet(good_index));
    TSS_temp = sum(error_benchmark_CENet(good_index));
    R_2_ADS(1,1) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_NORM(good_index));
    TSS_temp = sum(error_benchmark_DSC(good_index));
    R_2_ADS(2,1) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_EQ(good_index));
    TSS_temp = sum(error_benchmark_DSC(good_index));
    R_2_ADS(3,1) = 1 - SSE_temp/TSS_temp;

    norm_index = sort_index(part+1:2*part);
    SSE_temp = sum(error_CENet(norm_index));
    TSS_temp = sum(error_benchmark_CENet(norm_index));
    R_2_ADS(1,2) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_NORM(norm_index));
    TSS_temp = sum(error_benchmark_DSC(norm_index));
    R_2_ADS(2,2) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_EQ(norm_index));
    TSS_temp = sum(error_benchmark_DSC(norm_index));
    R_2_ADS(3,2) = 1 - SSE_temp/TSS_temp;

    bad_index =  sort_index(2*part+1:end);
    SSE_temp = sum(error_CENet(bad_index));
    TSS_temp = sum(error_benchmark_CENet(bad_index));
    R_2_ADS(1,3) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_NORM(bad_index));
    TSS_temp = sum(error_benchmark_DSC(bad_index));
    R_2_ADS(2,3) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_EQ(bad_index));
    TSS_temp = sum(error_benchmark_DSC(bad_index));
    R_2_ADS(3,3) = 1 - SSE_temp/TSS_temp;



    % sort by UNRATE
    [~,sort_index] = sort(econ_final(1:end-1,2),'ascend');
    part = floor(length(sort_index)/3);
    good_index = sort_index(1:part);
    SSE_temp = sum(error_CENet(good_index));
    TSS_temp = sum(error_benchmark_CENet(good_index));
    R_2_Unemploy(1,1) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_NORM(good_index));
    TSS_temp = sum(error_benchmark_DSC(good_index));
    R_2_Unemploy(2,1) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_EQ(good_index));
    TSS_temp = sum(error_benchmark_DSC(good_index));
    R_2_Unemploy(3,1) = 1 - SSE_temp/TSS_temp;

    norm_index = sort_index(part+1:2*part);
    SSE_temp = sum(error_CENet(norm_index));
    TSS_temp = sum(error_benchmark_CENet(norm_index));
    R_2_Unemploy(1,2) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_NORM(norm_index));
    TSS_temp = sum(error_benchmark_DSC(norm_index));
    R_2_Unemploy(2,2) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_EQ(norm_index));
    TSS_temp = sum(error_benchmark_DSC(norm_index));
    R_2_Unemploy(3,2) = 1 - SSE_temp/TSS_temp;

    bad_index =  sort_index(2*part+1:end);
    SSE_temp = sum(error_CENet(bad_index));
    TSS_temp = sum(error_benchmark_CENet(bad_index));
    R_2_Unemploy(1,3) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_NORM(bad_index));
    TSS_temp = sum(error_benchmark_DSC(bad_index));
    R_2_Unemploy(2,3) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_EQ(bad_index));
    TSS_temp = sum(error_benchmark_DSC(bad_index));
    R_2_Unemploy(3,3) = 1 - SSE_temp/TSS_temp;

    % sort by NBER
    
    good_index = find(econ_final(1:end-1,3) == 0);
    SSE_temp = sum(error_CENet(good_index));
    TSS_temp = sum(error_benchmark_CENet(good_index));
    R_2_NBER(1,1) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_NORM(good_index));
    TSS_temp = sum(error_benchmark_DSC(good_index));
    R_2_NBER(2,1) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_EQ(good_index));
    TSS_temp = sum(error_benchmark_DSC(good_index));
    R_2_NBER(3,1) = 1 - SSE_temp/TSS_temp;

    bad_index =  find(econ_final(1:end-1,3) == 1);
    SSE_temp = sum(error_CENet(bad_index));
    TSS_temp = sum(error_benchmark_CENet(bad_index));
    R_2_NBER(1,2) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_NORM(bad_index));
    TSS_temp = sum(error_benchmark_DSC(bad_index));
    R_2_NBER(2,2) = 1 - SSE_temp/TSS_temp;
    SSE_temp = sum(error_DSC_EQ(bad_index));
    TSS_temp = sum(error_benchmark_DSC(bad_index));
    R_2_NBER(3,2) = 1 - SSE_temp/TSS_temp;

   