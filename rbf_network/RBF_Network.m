clear
close all
T = 0.001; % 时间步长
x = @(t) sin(t); % 第一个输入函数
xt = [0.001:0.001:10]; %时间变量
hideNum = 5;      %隐含层节点数
alpha = 0.05;     %动量因子，反向传播训练用
yita  = 0.15;     %学习率，反向传播训练用
maxcount = 7000;   %学习次数，反向传播训练用
C = [-1,-0.5,0,0.5,1;-1,-0.5,0,0.5,1]';  %中心坐标向量，5个神经节点五个二维向量
dd = 3.*ones(1,5);    %高斯基函数宽度
y_real(1,1)=0;
for k = 2:1:10000
    y_real(1,k)=(x(k*T))^3 + y_real(1,k-1)/(1+(y_real(1,k-1))^2);     %离散函数的真实值求取
end
w = [0.5,0.5,0.5,0.5,0.5]';   %权值矩阵

%%开始训练
y_simu=zeros(1,10010); %第二个输入，通过RBF模拟的输出结果
%%求出隐含层b的输出
b = zeros(10000,hideNum);
for k = 1:1:10000
    x1_input(1,k) = x(k*T);    %rbf神经网络第一个输入
end
for i=1:1:maxcount
     for j=1:hideNum
         b(i,j)=exp(-((x1_input(1,i)-C(j,1))^2+(y_simu(1,i)-C(j,2))^2)/(2*dd(j)));
         y_simu(1,i+1)=y_simu(1,i+1)+b(i,j)*w(j,1);   %计算输出结果
     end
       %计算偏差
      error=y_real(1,i+1)-y_simu(1,i+1);
      for q=1:hideNum
          w(q,1)=w(q,1)+yita*error*b(i,q);   %%更新权重矩阵
      end
end
plot(xt(1:4000),y_real(1:4000),'linewidth',6);
hold on;
plot(xt(1:7000),y_simu(1:7000));
legend('实际数据','模型推测数据');
 
