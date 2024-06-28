% RBF Network Approximation using Gaussian Basis Function
% Created By Jiaxiao Shi on 2024/06/16. All rights reserved.

clear;
clc;
close all;

% 参数设定
ts=0.001;y_1=0;
for k=1:1:2000
    t(k)=k*ts;
    u(k)=0.5*sin(2*pi*t(k));
    y(k)=u(k)^3+y_1/(1+y_1^2);
    y_1=y(k);
end 
c=[-1.5 -0.5 0 0.5 1.5;-1.5 -0.5 0 0.5 1.5];b=0.0005*ones(5,1);
% c=0.1*[-1.5 -0.5 0 0.5 1.5;-1.5 -0.5 0 0.5 1.5];b=1.5*ones(5,1);
w=rand(5,1);a=0.05;n=0.15;x=[0;1];
w1=w;w2=w1;
for i=1:1:2000
    for j=1:1:5
        h(j)=exp(-norm(x-c(:,j))^2/(2*b(j)*b(j)));
    end
    ym(i)=w'*h';
    em(i)=y(i)-ym(i);
    w=w1+n*em(i)*h'+a*(w1-w2);
    w2=w1;w1=w;
    x(1)=0.5*sin(2*pi*i*ts);
    x(2)=y(i);
end 
figure(1);
plot(t,y,'g',t,ym,'k:','LineWidth',2);
xlabel('Time(s)','FontSize',15);ylabel('Output','FontSize',15);
legend('Actual','Prediction');