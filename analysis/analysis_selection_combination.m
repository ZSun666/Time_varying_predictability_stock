%% setup

date_start  = [1988,1]; %long sampledraw_figure = 0;

date_end = [2019,12];

date_subperiod ={[1988,2019],[2003,2019],[2012,2019]};
L_window = 240; % fixed window to compute y_mean and y_var;

%% load results
addpath('result')
addpath('function')
addpath('data')
clear beta
clear signal_index  
clear error_model
clear error_benchmark

predictors = readtable('new_dataset_full.xlsx');
var_name_1 = predictors.Properties.VariableNames(3:end);

% load individual predictor (First_stage)
    load first_stage.mat
    Result_first_stage = First_stage;

% load CENet
    load result_CENet
    Result_CENet = Result_fix; 

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
%% evaluate first stage
T_length_first_stage = length(Result_first_stage);

count = 0;

RSS = zeros(MK,1);
TSS = zeros(MK,1);

for t = 1:T_length_first_stage
    date_temp = Result_first_stage(t).date;
    y_predict(t,:) = Result_first_stage(t).y_predict;
    y_true(t,1) = Result_first_stage(t).y_true;
    y_to_t = Result_first_stage(t).y_to_t;
    y_mean_ind(t,1) = mean(y_to_t);
    nan_index = find(isnan(y_predict(t,:)));
    y_predict(t,nan_index) = y_mean_ind(t);
    if find(date_temp == date_range)
        start_index = max(1,t-60);
        % compute R_2 of each predictor at time t
        for m = 1:MK
            
            y_predict_temp = y_predict(start_index:t,m);
            y_true_temp = y_true(start_index:t);
            y_mean_ind_temp = y_mean_ind(start_index:t);
            obs_index = find(~isnan(y_predict_temp));
            TSS_temp = ((y_mean_ind_temp(obs_index) - y_true_temp(obs_index))'*(y_mean_ind_temp(obs_index) - y_true_temp(obs_index)));
            RSS_temp(m) = ((y_predict_temp(obs_index) - y_true_temp(obs_index))'*(y_predict_temp(obs_index) - y_true_temp(obs_index)));
            
            R2_temp(m) = 1-(RSS_temp(m))/TSS_temp;
        end
%         R2_temp = Result_first_stage(t).R2;
        R2_ind(count+1,:) = R2_temp;
        R2_temp(R2_temp<0.002) = 0;
        R2_temp(isnan(R2_temp)) = 0;
        beta_ind(count+1,:) = R2_temp./sum(R2_temp);
        concentrate_ind(count+1) = sum(beta_ind(count+1,:).^2);
%         concentrate_ind(count+1) = length(find(beta_ind(count+1,:)  ~=0));
        count = count+1;
    end

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

        concentrate_CENet(count+1) = sum(beta_CENet(count+1,:).^2);

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

        beta_DSC_ORIG(count+1,:) = Result_dsc(t).beta;
        beta_DSC_NORM(count+1,:) = Result_dsc(t).beta_norm;
        beta_DSC_EQ(count+1,:) = Result_dsc(t).beta_equal;

        concentrate_DSC_ORIG(count+1) = sum(beta_DSC_ORIG(count+1,:).^2);
        concentrate_DSC_NORM(count+1) = sum(beta_DSC_NORM(count+1,:).^2);
        concentrate_DSC_EQ(count+1) = sum(beta_DSC_EQ(count+1,:).^2);
        
        PIP_DSC(count+1,:) = Result_dsc(t).pi;
        if count > 0
            variation_DSC_ORIG(count+1) = sum((beta_DSC_ORIG(count+1,:) - beta_DSC_ORIG(count,:)).^2);
            variation_DSC_NORM(count+1) = sum((beta_DSC_NORM(count+1,:) - beta_DSC_NORM(count,:)).^2);
            variation_DSC_EQ(count+1) = sum((beta_DSC_EQ(count+1,:) - beta_DSC_EQ(count,:)).^2);
        else
            variation_DSC_ORIG(count+1) = 0;
            variation_DSC_NORM(count+1) = 0;
            variation_DSC_EQ(count+1) = 0;
        end

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

%% load the volatility of market
vol_var = readtable('econ_variable.xlsx'); % load econ var

vol_date = vol_var{:,1};
length_vol = length(vol_date);
tt = 1;
for t= 1:length_vol
    vol_date_t = num2str(vol_date(t));
    vol_date_t = datetime([str2num(vol_date_t(1:4)),str2num(vol_date_t(5:6)),1]);
    vol_date_save(t) = vol_date_t;
    vol_var_num(t,:) = vol_var{t,8};
    date_temp = vol_date_save(t);

    if find(date_temp == date_range)
        vol_final(tt,:) = vol_var_num(t,:); % get the variable during the evaluation period
        tt = tt+1;
    end
end


%% correlation between concentration of individual, CENet, DSC
[corr_concen,corr_concen_p] = corrcoef([concentrate_ind',concentrate_CENet',concentrate_DSC_EQ',concentrate_DSC_NORM']);

%% correlation between volatility and variation
[corr_variation,corr_variation_p] = corrcoef([vol_final(1:end-1),variation_DSC_NORM']);

%% Average PIP and correlation to individual R2
PIP_output = table;
mean_PIP = mean(PIP_DSC);
[sorted_mean_PIP,sorted_index] = sort(mean_PIP,'descend');
count_k = 1;
% only report the variable with PIP from 0.1 to 0.9
for k = 7:15
    PIP_output.var_name{count_k} = var_name_1{sorted_index(k)};
    PIP_output.mean_PIP{count_k} = mean(sorted_mean_PIP(k));
    [corr_PIP,corr_PIP_p] = corrcoef([R2_ind(:,sorted_index(k)),PIP_DSC(:,sorted_index(k))]);
    PIP_output.correlation{count_k} = corr_PIP(2,1);
    PIP_output.correlation_p{count_k} = corr_PIP_p(2,1);
    count_k = count_k+1;
end

%%  Figure: Performance of individual predictor
x(1) = date_start;
n_d = count;
for i = 2:n_d
    x(i) = x(i-1)+calmonths(1);
end
[~,rank_index] = sort(mean(R2_ind,1,'omitnan'),'descend');
select_index = rank_index(1:30);
figure
%     hold on
box on
xlabel = 1:length(x);
CustomXLabels = num2str(year(x)');

CustomXLabels(mod(xlabel-1,12) ~= 0,:) = " ";
h = heatmap(datestr(x),var_name_1(select_index),R2_ind(:,select_index)');
h.XDisplayLabels = CustomXLabels; % display xlabels
set(struct(h).NodeChildren(3), 'XTickLabelRotation',45); % rotate x label



%% Figure: Concentration of the model
figure('Position',[100,100,1500,900],'InvertHardcopy','off','Color',[1 1 1]);
n_d = count;
x(1) = date_start;
for i = 2:n_d
    x(i) = x(i-1)+calmonths(1);
end
y_bench = zeros(n_d,1);

y_1 = concentrate_CENet;
y_2 = concentrate_DSC_NORM;
y_3 = concentrate_DSC_EQ;




hold on
box on
plot(x,y_1,'LineWidth',2);
plot(x,y_2,'LineWidth',2);
plot(x,y_3,'LineWidth',2);
%     plot(x,y_3,'LineWidth',2);
%     plot(x,y_4,'LineWidth',2);


legend('CENet','DSC-NORM','DSC-EQ','AutoUpdate','off','Location','northeast','Fontweight','bold')
%     plot(x,y_bench,'LineWidth',2,'Color','black','LineStyle','- -');

xlim([x(1) x(end)]);   
datetick('x','mmm-yy','keeplimits','keepticks')
ylim([0,0.3])
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

%%  Figure:  Variation in the combination weights
figure('Position',[100,100,1500,900],'InvertHardcopy','off','Color',[1 1 1]);
n_d = count;
x(1) = date_start;
for i = 2:n_d
    x(i) = x(i-1)+calmonths(1);
end
y_bench = zeros(n_d,1);
x = x(2:end);
y_0 = variation_DSC_NORM(2:end);




hold on
box on
plot(x,y_0,'LineWidth',2);



legend('DSC-NORM','AutoUpdate','off','Location','northeast','Fontweight','bold')

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

%%  Figure: PIP over time
figure('Position',[100,100,1500,900],'InvertHardcopy','off','Color',[1 1 1]);
n_d = count;
x(1) = date_start;
for i = 2:n_d
    x(i) = x(i-1)+calmonths(1);
end
count_k = 1;
for k = 7:15
    subplot(3,3,count_k)
    plot(x,PIP_DSC(:,sorted_index(k)))
    var_name = var_name_1{sorted_index(k)};
    title(var_name)
    ylim([0,1])
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
    count_k = count_k+1;
end
