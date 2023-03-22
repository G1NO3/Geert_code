n = zeros(31);
m = zeros(31);
n(1) = 400;
m(1) = 120;
for i=2:31
    n(i) = n(i-1)*1.1;
    m(i) = m(i-1)*1.2;
end
figure;
hold on;
plot(log(n));
plot(log(m));
legend({'in me','in my assistant'},'Location','northwest');
xlabel('days');
ylabel('number of parasites');

