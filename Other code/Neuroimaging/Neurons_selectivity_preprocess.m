data = importdata('R15N111_Raw.mat');
fs = 24414.0625;
dt = 1/fs;
t = 0:dt:(length(data)-1)*dt;
t = t';
figure;

% plot(t, data);
% xlabel('time');
% ylabel('response');
% xlim([0 (length(data)-1)*dt]);

%Spiking matrix
threshold = 40;
spiking_matrix = zeros(50,1) + threshold;
timestamp_total = zeros(1,1);
i = 1;
while i<=length(data)
    if data(i)>threshold
        spiking_pattern = data(i-20:i+29);
        i = i+29;
        spiking_matrix = [spiking_matrix,spiking_pattern];
        timestamp_total = [timestamp_total;i];
    end
    i=i+1;
end

SM = double(spiking_matrix');
[C,score,latent] = pca(SM);
subplot(3,1,1);
plot(C(:,(1:2)));
xlabel('point');
ylabel('response');
title('1-1 Principal Component of spikes');
subplot(3,1,2);
plot(cumsum(latent/sum(latent)));
xlabel('i-th component');
ylabel('Variance');
title('1-2 Cumulated Explained Ratio of different PCs');
subplot(3,1,3);
scatter(score(:,1),score(:,2),".");
xlabel('PC1');
ylabel('PC2');
title('1-3 Distribution of Spikes with respect to PC1&PC2 (Red:Decision Boundary)')

line([0,0],[-400,400],'Color','red','Linestyle','-','Linewidth',3);
index_1 = find(score(:,1)<0);
tstmp_1 = timestamp_total(index_1)*dt;
index_2 = find(score(:,1)>=0);
tstmp_2 = timestamp_total(index_2)*dt;

save('timestamp_1.mat','tstmp_1');
save('timestamp_2.mat','tstmp_2');






