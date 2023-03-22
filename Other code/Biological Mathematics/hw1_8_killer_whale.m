A = [0 .0043 .1132 0;
     .9775 .9111 0 0;
     0 .0736 .9534 0;
     0 0 .0452 .9804];
[eigvec,eigval] = eig(A);
[d,ind] = sort(diag(eigval));
eigvec = eigvec(:,ind);
stable_distribution = eigvec(:,end);
lambda = d(end);

x = zeros(4,51);
x(:,1) = [10;60;110;70];
sample = 250*eye(4);
x(:,1) = sample(:,1);
for i=2:51
    x(:,i) = A*x(:,i-1);
end
x = x(:,2:end);
figure;
subplot(2,2,1);
hold on;
plot(x(1,:));
plot(x(2,:));
plot(x(3,:));
plot(x(4,:));
legend({'age1','age2','age3','age4'},'Location','northwest');
title('population dynamics for the next 50 years');

N = sum(x,1);
growth = N./[1,N(1:end-1)];
growth = growth(2:end);
proportion = x./repmat(N,4,1);
subplot(2,2,2);
plot(N);
title('total population size');
subplot(2,2,3);
plot(growth);
title('annual population growth rate');
subplot(2,2,4);
hold on;
plot(proportion(1,:));
plot(proportion(2,:));
plot(proportion(3,:));
plot(proportion(4,:));
title('proportion of individuals');
legend({'age1','age2','age3','age4'},'Location','east');

%test
[eigvec_left,eigval_left] = eig(A');
[d_left,ind_left] = sort(diag(eigval_left));
eigvec_left = eigvec_left(:,ind_left);
v = eigvec_left(:,end);
w = stable_distribution;

sensitivity = (v*w')./(v'*w);
figure;
heatmap(sensitivity);
title('sensitivity matrix');



h=[0 -3 -3 -3]';
A_prime = [A,h];
row = [0 0 0 0 1];
A_prime = [A_prime; row];
x_prime = zeros(5,51);
x_prime(:,1) = round([w.*250;1]);
for i = 2:51
    x_prime(:,i) = A_prime * x_prime(:,i-1);
end
figure;
N_prime = sum(x_prime,1);
plot(N_prime);

