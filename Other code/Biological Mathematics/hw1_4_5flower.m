n=zeros(3,50);
L=[0 1 5;
   .5 0 0;
   0 .25 0];
n(1,1)=1;
for i =2:50
    n(:,i) = L*n(:,i-1);
end
N = sum(n,1);
w = n./repmat(N,3,1);
figure;
subplot(1,3,1);
plot(log(N));
xlabel('time');
ylabel('log population size');
subplot(1,3,2);
hold on;
plot(w(1,:));
plot(w(2,:));
plot(w(3,:));
legend({'age-1','age-2', 'age-3'});
ylabel('ratio');
xlabel('time');
subplot(1,3,3);
lambda = fsolve(@(x) 0.5*x.^(-2)+0.5.*0.25.*5.*x.^(-3)-1,2);
l_change = [n(1,2:end),0]./n(1,:);
l_change = l_change(1:end-1);
hold on;
plot(l_change,'LineWidth',2);
line([1 50],[lambda lambda],'Color','red','Linewidth',1);
title('rate of population growth');
xlabel('time');