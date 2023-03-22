A=[1 1;1 0];
x=zeros(2,100);
x(:,1)=[1;1];
for i=1:99
    x(:,i+1) = A*x(:,i);
end 
r=x(1,:)./x(2,:);
figure;
plot(r);



