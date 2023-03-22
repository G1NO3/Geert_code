tstmp_1 = importdata('timestamp_1.mat');
tstmp_2 = importdata('timestamp_2.mat');
data = importdata('R15N111_Raw.mat');
stimulus = struct2cell(importdata('R15N111_Stimulus.mat'));
fs = 24414.0625;
dt = 0.25;
[freqs,levels,sti_onset] = stimulus{:};
n_freq = 20;
FRA_1 = zeros(9,n_freq);
FRA_2 = zeros(9,n_freq);
level_axis = (0:5:40);
d_freq = (max(freqs)-min(freqs))/n_freq;
freq_axis = (min(freqs):d_freq:max(freqs)-d_freq);

for i=1:length(sti_onset)
    if freqs(i)<max(freqs)
        i_freq = floor((freqs(i)-min(freqs))/d_freq)+1;
    else
        i_freq = n_freq;
    end
    i_level = levels(i)/5+1;
    t_start = (i-1)*dt;
    spike_counts_1 = size(tstmp_1(tstmp_1>=t_start & tstmp_1<t_start+dt),1);
    spike_counts_2 = size(tstmp_2(tstmp_2>=t_start & tstmp_2<t_start+dt),1);
    FRA_1(i_level,i_freq) = spike_counts_1;
    FRA_2(i_level,i_freq) = spike_counts_2;
end

figure;
subplot(2,4,1);
heatmap(freq_axis,level_axis,FRA_1);
title('FRA_1');
xlabel('frequency/Hz');
ylabel('level');
subplot(2,4,2);
heatmap(freq_axis,level_axis,FRA_2);
title('FRA_2');
xlabel('frequency/Hz');
ylabel('level');


[maxl_1,maxf_1] = find(FRA_1==max(FRA_1,[],'all'));
[maxl_2,maxf_2] = find(FRA_2==max(FRA_2,[],'all'));
needed_points = ceil(fs*0.1);
voltage_1 = zeros(needed_points,1);
voltage_2 = zeros(needed_points,1);
for i=1:length(sti_onset)
    if freqs(i)<max(freqs)
        i_freq = floor((freqs(i)-min(freqs))/d_freq)+1;
    else
        i_freq = n_freq;
    end
    i_level = levels(i)/5+1;
    if i_freq==maxf_1 && i_level==maxl_1
        t_start = (i-1)*dt;
        i_start = floor(t_start*fs);
        voltage_1 = [voltage_1,data(i_start:i_start+needed_points-1)];
    end
    if i_freq==maxf_2 && i_level==maxl_2
        t_start = (i-1)*dt;
        i_start = floor(t_start*fs);
        voltage_2 = [voltage_2,data(i_start:i_start+needed_points-1)];
    end
end
tx = (0:1/fs:(needed_points-1)/fs);
tx = tx';
voltage_1 = voltage_1(:,2:end);
voltage_2 = voltage_2(:,2:end);
subplot(2,4,3);
plot(tx,voltage_1);
xlabel('time since feauture stimulus/s');
ylabel('response');
title('Response_1 to Feature Freq&Level');
subplot(2,4,4);
plot(tx,voltage_2);
xlabel('time since feauture stimulus/s');
ylabel('response');
title('Response_2 to Feature Freq&Level');



bins=20;
subplot(2,4,5);
histogram(diff(tstmp_1),bins);
title('Overall ISI histogram of N1');
xlabel('ISI/s');
ylabel('Counts');
subplot(2,4,6);
histogram(diff(tstmp_2),bins);
title('Overall ISI histogram of N2');
xlabel('ISI/s');
ylabel('Counts');

ISI_1 = zeros(1,1);
ISI_2 = zeros(1,1);
for i=1:length(sti_onset)
    if freqs(i)<max(freqs)
        i_freq = floor((freqs(i)-min(freqs))/d_freq)+1;
    else
        i_freq = n_freq;
    end
    i_level = levels(i)/5+1;
    if i_freq<=maxf_1+1 && i_freq>=maxf_1-1 && i_level>=maxl_1-1 && i_level<=maxl_1+1
        t_start = floor(i*dt);
        t_end = ceil(i*dt+0.1);
        ISI_1 = [ISI_1;diff(tstmp_1(tstmp_1>=t_start & tstmp_1<=t_end))];
    end

    if i_freq<=maxf_2+1 && i_freq>=maxf_2-1 && i_level>=maxl_2-1 && i_level<=maxl_2+1
        t_start = floor(i*dt);
        t_end = ceil(i*dt+0.1);
        ISI_2 = [ISI_2;diff(tstmp_2(tstmp_2>=t_start & tstmp_2<=t_end))];
    end

end
ISI_1 = ISI_1(2:end);
ISI_2 = ISI_2(2:end);
subplot(2,4,7);
histogram(ISI_1,bins);
title('ISI histogram of FRA of N1');
xlabel('ISI/s');
ylabel('Counts');
subplot(2,4,8);
histogram(ISI_2,bins);
title('ISI histogram of FRA of N2');
xlabel('ISI/s');
ylabel('Counts');
