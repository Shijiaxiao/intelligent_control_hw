ts=0.001;y_1=0;
for k=1:1:10000
    t(k)=k*ts;
    u(k)=sin(t(k));
    y(k)=u(k)^3+y_1/(1+y_1^2);
    y_1=y(k);
end
c=[-1 -0.5 0 0.5 1;-1 -0.5 0 0.5 1];b=3*ones(5,1);
w=rand(5,1);a=0.05;n=0.15;x=[0;1];
w1=w;w2=w1;
for i=1:1:10000
    for j=1:1:5
        h(j)=exp(-norm(x-c(:,j))^2/(2*b(j)*b(j)));
    end
    ym(i)=w'*h';
    em(i)=y(i)-ym(i);
    w=w1+n*em(i)*h'+a*(w1-w2);
    w2=w1;w1=w;
    x(1)=sin(i*ts);
    x(2)=y(i);
end
figure(1);
subplot(2,1,1);
plot(t,y,'r',t,ym,'k:','LineWidth',2);
xlabel('t(s)','FontSize',18);ylabel('y/ym','FontSize',18);
legend('理想曲线','逼近曲线');
subplot(2,1,2);
plot(t,y-ym,'k','LineWidth',2);
xlabel('t(s)','FontSize',18);ylabel('error','FontSize',18);
