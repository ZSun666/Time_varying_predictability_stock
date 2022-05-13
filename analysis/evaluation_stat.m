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
MSFE = zeros(10,3);
pvalue = zeros(10,3);
R2 = zeros(10,3);
%% evaluate pca    
T_length_pca = length(Result_pca);

count = 0;
RSS = 0;
TSS = 0;

for t = 1:T_length_pca  
    date_temp = Result_pca(t).date;

    if find(date_temp == date_range)        
        beta_pca(count+1,:) = Result_pca(t).beta;

 
        y_predict_pca(count+1,:) = Result_pca(t).y_predict;
        y_true_pca(count+1) = Result_pca(t).y_true(end);
        y_to_t = Result_pca(t).y_to_t;
        y_mean = mean(y_to_t(1:end));

        error_date_pca(count+1) = date_temp;%+calmonths(1);
        error_pca(count+1) = (y_predict_pca(count+1)-y_true_pca(count+1))^2;
   
        error_benchmark_pca(count+1) = (y_mean - y_true_pca(count+1))^2;
        RSS = RSS + error_pca(count+1);
        TSS = TSS + error_benchmark_pca(count+1);
        Diff_CSE_pca(count+1) = (RSS) - (TSS);
        R_squared_pca(count+1) = 1- RSS/TSS;
   
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
    RSS_temp = sum(error_pca(index_start:index_end));
    TSS_temp = sum(error_benchmark_pca(index_start:index_end));
    R2(1,ii) = 1-RSS_temp./TSS_temp;
    MSFE(1,ii) = sqrt(mean(error_pca(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_pca(index_start:index_end))',sqrt(error_benchmark_pca(index_start:index_end))',1);
    pvalue(1,ii) = 1-mvncdf(abs(DM_test_temp));
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
        beta_ENet(count+1,:) = Result_ENet(t).beta;

 
        y_predict_ENet(count+1,:) = Result_ENet(t).y_predict;
        y_true_ENet(count+1) = Result_ENet(t).y_true(end);
        y_to_t = Result_ENet(t).y_to_t;
        y_mean = mean(y_to_t(1:end));

        error_date_ENet(count+1) = date_temp;%+calmonths(1);
        error_ENet(count+1) = (y_predict_ENet(count+1)-y_true_ENet(count+1))^2;
   
        error_benchmark_ENet(count+1) = (y_mean - y_true_ENet(count+1))^2;
        RSS = RSS + error_ENet(count+1);
        TSS = TSS + error_benchmark_ENet(count+1);
        Diff_CSE_ENet(count+1) = (RSS) - (TSS);
%             Diff_SFE(count+1) = (RSS) - (TSS);
        R_squared_ENet(count+1) = 1- RSS/TSS;
   
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
    RSS_temp = sum(error_ENet(index_start:index_end));
    TSS_temp = sum(error_benchmark_ENet(index_start:index_end));
    R2(2,ii) = 1-RSS_temp./TSS_temp;
    MSFE(2,ii) = sqrt(mean(error_ENet(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_ENet(index_start:index_end))',sqrt(error_benchmark_ENet(index_start:index_end))',1);
    pvalue(2,ii) = 1-mvncdf(abs(DM_test_temp));
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
   
    R2(3,ii) = 1-RSS_temp./TSS_temp;
    MSFE(3,ii) = sqrt(mean(error_CENet(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_CENet(index_start:index_end))',sqrt(error_benchmark_CENet(index_start:index_end))',1);
    pvalue(3,ii) = 1-mvncdf(abs(DM_test_temp));
end


%% evaluate GBRT
T_length_GBRT = length(Result_GBRT);

count = 0;
RSS = 0;
TSS = 0;

for t = 1:T_length_GBRT
    % datetime    
    date_temp = Result_GBRT(t).date;

    if find(date_temp == date_range)        

        y_predict_GBRT(count+1,:) = Result_GBRT(t).y_predict;
        y_true_GBRT(count+1) = Result_GBRT(t).y_true(end);
        y_to_t = Result_GBRT(t).y_to_t;
        y_mean = mean(y_to_t(1:end));

        error_date_GBRT(count+1) = date_temp;%+calmonths(1);
        error_GBRT(count+1) = (y_predict_GBRT(count+1)-y_true_GBRT(count+1))^2;
   
        error_benchmark_GBRT(count+1) = (y_mean - y_true_GBRT(count+1))^2;
        RSS = RSS + error_GBRT(count+1);
        TSS = TSS + error_benchmark_GBRT(count+1);
        Diff_CSE_GBRT(count+1) = (RSS) - (TSS);
        R_squared_GBRT(count+1) = 1- RSS/TSS;
   
        count = count + 1;

    end  
end

% compute MSFE, pvalue,R2 during each subperiods
for ii = 1:length(date_subperiod)
    date_subrange = date_subperiod{ii};
    date_start_sub = datetime([date_subrange(1),1,1]);
    date_end_sub = datetime([date_subrange(2),11,1]);
    index_start = find(date_start_sub == error_date_GBRT);
    index_end = find(date_end_sub == error_date_GBRT);
    RSS_temp = sum(error_GBRT(index_start:index_end));
    TSS_temp = sum(error_benchmark_GBRT(index_start:index_end));
   
    R2(4,ii) = 1-RSS_temp./TSS_temp;
    MSFE(4,ii) = sqrt(mean(error_GBRT(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_GBRT(index_start:index_end))',sqrt(error_benchmark_GBRT(index_start:index_end))',1);
    pvalue(4,ii) = 1-mvncdf(abs(DM_test_temp));
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

        y_predict_RF(count+1,:) = Result_RF(t).y_predict;
        y_true_RF(count+1) = Result_RF(t).y_true(end);
        y_to_t = Result_RF(t).y_to_t;
        y_mean = mean(y_to_t(1:end));

        error_date_RF(count+1) = date_temp;%+calmonths(1);
        error_RF(count+1) = (y_predict_RF(count+1)-y_true_RF(count+1))^2;
   
        error_benchmark_RF(count+1) = (y_mean - y_true_RF(count+1))^2;
        RSS = RSS + error_RF(count+1);
        TSS = TSS + error_benchmark_RF(count+1);
        Diff_CSE_RF(count+1) = (RSS) - (TSS);
        R_squared_RF(count+1) = 1- RSS/TSS;
   
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
    RSS_temp = sum(error_RF(index_start:index_end));
    TSS_temp = sum(error_benchmark_RF(index_start:index_end));
   
    R2(5,ii) = 1-RSS_temp./TSS_temp;
    MSFE(5,ii) = sqrt(mean(error_RF(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_RF(index_start:index_end))',sqrt(error_benchmark_RF(index_start:index_end))',1);
    pvalue(5,ii) = 1-mvncdf(abs(DM_test_temp));
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

        y_predict_NN1(count+1,:) = Result_NN1(t).y_predict;
        y_true_NN1(count+1) = Result_NN1(t).y_true(end);
        y_to_t = Result_NN1(t).y_to_t;
        y_mean = mean(y_to_t(1:end));

        error_date_NN1(count+1) = date_temp;%+calmonths(1);
        error_NN1(count+1) = (y_predict_NN1(count+1)-y_true_NN1(count+1))^2;
   
        error_benchmark_NN1(count+1) = (y_mean - y_true_NN1(count+1))^2;
        RSS = RSS + error_NN1(count+1);
        TSS = TSS + error_benchmark_NN1(count+1);
        Diff_CSE_NN1(count+1) = (RSS) - (TSS);
        R_squared_NN1(count+1) = 1- RSS/TSS;
   
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
    RSS_temp = sum(error_NN1(index_start:index_end));
    TSS_temp = sum(error_benchmark_NN1(index_start:index_end));
   
    R2(6,ii) = 1-RSS_temp./TSS_temp;
    MSFE(6,ii) = sqrt(mean(error_NN1(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_NN1(index_start:index_end))',sqrt(error_benchmark_NN1(index_start:index_end))',1);
    pvalue(6,ii) = 1-mvncdf(abs(DM_test_temp));
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

        y_predict_NN2(count+1,:) = Result_NN2(t).y_predict;
        y_true_NN2(count+1) = Result_NN2(t).y_true(end);
        y_to_t = Result_NN2(t).y_to_t;
        y_mean = mean(y_to_t(1:end));

        error_date_NN2(count+1) = date_temp;%+calmonths(1);
        error_NN2(count+1) = (y_predict_NN2(count+1)-y_true_NN2(count+1))^2;
   
        error_benchmark_NN2(count+1) = (y_mean - y_true_NN2(count+1))^2;
        RSS = RSS + error_NN2(count+1);
        TSS = TSS + error_benchmark_NN2(count+1);
        Diff_CSE_NN2(count+1) = (RSS) - (TSS);
        R_squared_NN2(count+1) = 1- RSS/TSS;
   
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
    RSS_temp = sum(error_NN2(index_start:index_end));
    TSS_temp = sum(error_benchmark_NN2(index_start:index_end));
   
    R2(7,ii) = 1-RSS_temp./TSS_temp;
    MSFE(7,ii) = sqrt(mean(error_NN2(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_NN2(index_start:index_end))',sqrt(error_benchmark_NN2(index_start:index_end))',1);
    pvalue(7,ii) = 1-mvncdf(abs(DM_test_temp));
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
    R2(8,ii) = 1-RSS_temp./TSS_temp;
    MSFE(8,ii) = sqrt(mean(error_DSC_ORIG(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_DSC_ORIG(index_start:index_end))',sqrt(error_benchmark_DSC(index_start:index_end))',1);
    pvalue(8,ii) = 1-mvncdf(abs(DM_test_temp));

    % DSC_NORM
    RSS_temp = sum(error_DSC_NORM(index_start:index_end));
    R2(9,ii) = 1-RSS_temp./TSS_temp;
    MSFE(9,ii) = sqrt(mean(error_DSC_NORM(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_DSC_NORM(index_start:index_end))',sqrt(error_benchmark_DSC(index_start:index_end))',1);
    pvalue(9,ii) = 1-mvncdf(abs(DM_test_temp));

    % DSC_EQ
    RSS_temp = sum(error_DSC_EQ(index_start:index_end));
    R2(10,ii) = 1-RSS_temp./TSS_temp;
    MSFE(10,ii) = sqrt(mean(error_DSC_EQ(index_start:index_end)));
    DM_test_temp = dmtest(sqrt(error_DSC_EQ(index_start:index_end))',sqrt(error_benchmark_DSC(index_start:index_end))',1);
    pvalue(10,ii) = 1-mvncdf(abs(DM_test_temp));
end
%% statististical Evaluation
stat_output = table;
model_name = {'PCA','ENet','CENet','GBRT','RF','NN1','NN2','DSC-ORIG','DSC-NORM','DSC-EQ'};
for i_model = 1:10
    stat_output.model_name{i_model}= model_name{i_model};
    % subperiod 1988-2019
    stat_output.MSFE_1{i_model}= MSFE(i_model,1);
    stat_output.pvalue_1{i_model}= pvalue(i_model,1);
    stat_output.R2_1{i_model}= R2(i_model,1);
    % subperiod 2003-2019
    stat_output.MSFE_2{i_model}= MSFE(i_model,2);
    stat_output.pvalue_2{i_model}= pvalue(i_model,2);
    stat_output.R2_2{i_model}= R2(i_model,2);
    % subperiod 2012-2019
    stat_output.MSFE_3{i_model}= MSFE(i_model,3);
    stat_output.pvalue_3{i_model}= pvalue(i_model,3);
    stat_output.R2_3{i_model}= R2(i_model,3);
end

%% Figure: Differences in Cumulative Forecast Error

figure('Position',[100,100,1500,900],'InvertHardcopy','off','Color',[1 1 1]);
n_d = count;
x(1) = date_start;
for i = 2:n_d
    x(i) = x(i-1)+calmonths(1);
end
y_bench = zeros(n_d,1);

y_0 = Diff_CSE_DSC;  
y_1 = Diff_CSE_pca;
y_2 = Diff_CSE_ENet;
y_3 = Diff_CSE_CENet;
y_4 = Diff_CSE_RF;
y_5 = Diff_CSE_NN1;
%     


hold on
box on
% plot figures
plot(x,y_0,'LineWidth',2);
plot(x,y_1,'LineWidth',2);
plot(x,y_2,'LineWidth',2);
plot(x,y_3,'LineWidth',2);
plot(x,y_4,'LineWidth',2);
plot(x,y_5,'LineWidth',2);

% plot legend
legend('DSC-NORM','PCA','ENet','CENet','RF','NN1','AutoUpdate','off','Location','northeast','Fontweight','bold')
plot(x,y_bench,'LineWidth',2,'Color','black','LineStyle','- -');

% time x-axis
xlim([x(1) x(end)]);   
datetick('x','QQ-yy','keeplimits','keepticks')

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


