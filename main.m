%% This is the main file in the project.
%% Use this code to generate the predictions by each method, and save the predictions in the folder /result.

addpath('main_algorithms')

%% Prediction by individual predictor, used for model combination methods (CENet and DSC)
first_stage

%% Principal Component Regression (PCR)
main_pca

%% Elastic Net (ENet)
main_eNet

%% Combination ENet (CENet)
main_CEnet

%% Random Forest
main_RF

%% Neural Network
main_NN

%% DSC
main_dsc