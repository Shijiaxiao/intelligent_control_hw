% RBF Network Approximation using Gaussian Basis Function
% Created By Jiaxiao Shi on 2024/06/16. All rights reserved.

clear;
clc;
close all;

% 参数设定
time_step = 0.001;
input_func = @(t) sin(t); 
time_var = [0.001:0.001:10]; 
hidden_nodes = 5; 
momentum_factor = 0.05;
learning_rate = 0.15; 
max_iterations = 5000;

% 中心坐标向量
centers = [-1,-0.5,0,0.5,1; -1,-0.5,0,0.5,1]'; 

% 高斯基函数宽度
widths = 3.*ones(1,5); 

% 权值矩阵
weights = [0.5,0.5,0.5,0.5,0.5]';   

% 离散函数取值
y_actual(1,1) = 0;
for k = 2:1:10000
    y_actual(1,k) = (input_func(k*time_step))^3 + y_actual(1,k-1)/(1+(y_actual(1,k-1))^2); 
end

% 训练RBF神经网络
y_predicted = zeros(1,10010);
b = zeros(10000, hidden_nodes);
for k = 1:1:10000
    input_rbf(1,k) = input_func(k*time_step);   
end
for i = 1:1:max_iterations-1
    for j = 1:hidden_nodes
        b(i,j) = exp(-((input_rbf(1,i)-centers(j,1))^2+(y_predicted(1,i)-centers(j,2))^2)/(2*widths(j)));
        y_predicted(1,i+1) = y_predicted(1,i+1) + b(i,j)*weights(j,1); 
    end
    error = y_actual(1,i+1) - y_predicted(1,i+1);
    for q = 1:hidden_nodes
        weights(q,1) = weights(q,1) + learning_rate*error*b(i,q);
    end
end

plot(time_var(1:4000), y_actual(1:4000), 'linewidth', 2);
hold on;
plot(time_var(1:7000), y_predicted(1:7000));
legend('Actual Data', 'Model Prediction');
