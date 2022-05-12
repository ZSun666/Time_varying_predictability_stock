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

%% date range
date_start = datetime([date_start,1]);
date_end = datetime([date_end,1]);
date_range = date_start:calmonths(1):date_end;

stat_output = table;
for m_data = 1:5
    
    

% load CENet
    load_name = strcat('result_robust/result_CENet_robust_',num2str(m_data));
    load(load_name)
    Result_CENet = Result_fix; 

% load dsc
    load_name = strcat('result_robust/result_dsc_robust_',num2str(m_data));
    load(load_name)
    Result_dsc = Result_DSC;    


 
    
    MK = size(Result_dsc(1).x_predict,2);
    
    
    
    %% temp matrix
    MSFE = zeros(20,3);
    pvalue = zeros(20,3);
    R2 = zeros(20,3);
  
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
    RSS = 0;
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
            RSS = RSS + error_DSC_NORM(count+1);
            TSS = TSS + error_benchmark_DSC(count+1);
            Diff_CSE_DSC(count+1) = (RSS) - (TSS);
            R_squared_DSC(count+1) = 1- RSS/TSS;
       
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
    %% statististical Evaluation
    
    model_name = {'CENet','DSC-ORIG','DSC-NORM','DSC-EQ'};
    for i_model = 1:4
        index_temp = (m_data-1)*4+i_model;
        stat_output.model_name{index_temp}= model_name{i_model};
        % subperiod 1988-2019
        stat_output.MSFE_1{index_temp}= MSFE(i_model,1);
        stat_output.pvalue_1{index_temp}= pvalue(i_model,1);
        stat_output.R2_1{index_temp}= R2(i_model,1);
        % subperiod 2003-2019
        stat_output.MSFE_2{index_temp}= MSFE(i_model,2);
        stat_output.pvalue_2{index_temp}= pvalue(i_model,2);
        stat_output.R2_2{index_temp}= R2(i_model,2);
        % subperiod 2012-2019
        stat_output.MSFE_3{index_temp}= MSFE(i_model,3);
        stat_output.pvalue_3{index_temp}= pvalue(i_model,3);
        stat_output.R2_3{index_temp}= R2(i_model,3);
    end

end