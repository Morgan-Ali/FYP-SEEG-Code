clear;

load preprocessing1.mat;

g = Datacell{1,1}(:,1);
x = Datacell{1,2}(:,1);

fs = 1000;

y = cat(1,g,x);
% Define the target sampling frequency
new_fs = 500; % Hz

y_new = resample(y, new_fs, fs);
% Determine the downsampling factor

% Plot the original and downsampled signals
t_orig = (0:length(y)-1)/fs;
t_down = (0:length(y_new)-1)/new_fs;

figure;
subplot(3,1,1);
plot(t_orig, y);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Signal (Fs = 1000 Hz)');

subplot(3,1,2);
plot(t_down, y_new);
xlabel('Time (s)');
ylabel('Amplitude');
title(sprintf('Downsampled Signal (Fs = 500 Hz)'));

f1 = 0.5;  % Lower cutoff frequency in Hz
f2 = 200;  % Upper cutoff frequency in Hz
order = 4; % Filter order

% Design the Butterworth filter
[b, a] = butter(order, [f1/(new_fs/2), f2/(new_fs/2)], 'bandpass');

% Apply the filter to the signal
y_filtered = filtfilt(b, a, y_new);

% Define the notch filter parameters
f0 = 50;  % Notch frequency in Hz
Q = 30;   % Quality factor
bw = (f0/(new_fs/2))/Q; % Bandwidth of the notch filter

% Design the notch filter
[b2, a2] = iirnotch(bw, f0/(new_fs/2));

% Apply the filter to the signal
y_double_filtered = filtfilt(b2, a2, y_filtered);

subplot(3,1,3);
plot(t_down, y_filtered);
xlabel('Time (s)');
ylabel('Amplitude');
title(sprintf('Bandpass Filtered Signal'));