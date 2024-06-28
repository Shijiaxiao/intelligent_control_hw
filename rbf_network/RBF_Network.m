clear
close all
T = 0.001; % ʱ�䲽��
x = @(t) sin(t); % ��һ�����뺯��
xt = [0.001:0.001:10]; %ʱ�����
hideNum = 5;      %������ڵ���
alpha = 0.05;     %�������ӣ����򴫲�ѵ����
yita  = 0.15;     %ѧϰ�ʣ����򴫲�ѵ����
maxcount = 7000;   %ѧϰ���������򴫲�ѵ����
C = [-1,-0.5,0,0.5,1;-1,-0.5,0,0.5,1]';  %��������������5���񾭽ڵ������ά����
dd = 3.*ones(1,5);    %��˹���������
y_real(1,1)=0;
for k = 2:1:10000
    y_real(1,k)=(x(k*T))^3 + y_real(1,k-1)/(1+(y_real(1,k-1))^2);     %��ɢ��������ʵֵ��ȡ
end
w = [0.5,0.5,0.5,0.5,0.5]';   %Ȩֵ����

%%��ʼѵ��
y_simu=zeros(1,10010); %�ڶ������룬ͨ��RBFģ���������
%%���������b�����
b = zeros(10000,hideNum);
for k = 1:1:10000
    x1_input(1,k) = x(k*T);    %rbf�������һ������
end
for i=1:1:maxcount
     for j=1:hideNum
         b(i,j)=exp(-((x1_input(1,i)-C(j,1))^2+(y_simu(1,i)-C(j,2))^2)/(2*dd(j)));
         y_simu(1,i+1)=y_simu(1,i+1)+b(i,j)*w(j,1);   %����������
     end
       %����ƫ��
      error=y_real(1,i+1)-y_simu(1,i+1);
      for q=1:hideNum
          w(q,1)=w(q,1)+yita*error*b(i,q);   %%����Ȩ�ؾ���
      end
end
plot(xt(1:4000),y_real(1:4000),'linewidth',6);
hold on;
plot(xt(1:7000),y_simu(1:7000));
legend('ʵ������','ģ���Ʋ�����');
 
