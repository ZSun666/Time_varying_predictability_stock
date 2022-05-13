# Time_varying_predictability_stock
This is the replication code for 'Time-varying predictability of stock return with high-dimensional data'.

In this paper, we propose a particle filter with variational Bayes (PF-VB) algorithm to estimate the time-varying parameters predictive model with high-dimensional dataset. The code of proposed method is saved on "/main_algorithms/main_dsc"

We predict the S&P 500 stock return with proposed method, and compare it with other methods.

How to replicate the results by the codes:

1. Run the file "main.m" to generate the predictions by different methods. All the methods are saved on "/main_algorithms/". The results are saved on "/result/".

2. Run the file "analysis.m" to generate the tables and figures. This file will load all the results saved on "/result/", and make sure the save/load name of each result file is consistent. 

3. All the figures are saved on "/Figure/".


