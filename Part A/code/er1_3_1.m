clear all;
close all;
image = imread('../images/board.png');
image = im2double(image);

%%%%%%%%%%%%%%%%%
% Calculate the mean and variance of the image
mean_val = mean(image(:));
variance_val = var(image(:));

% Desired SNR in dB
desired_snr_db = 15;

% Convert SNR from dB to linear scale
desired_snr_linear = 10^(desired_snr_db/10);

% Calculate the noise variance required for desired SNR
desired_noise_variance = variance_val / desired_snr_linear;

% Add white Gaussian noise with the calculated variance
noisy_image = imnoise(image, 'gaussian', 0, desired_noise_variance);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
signal_power = mean(image(:).^2); % Calculate the power of the image signal

% Calculate the desired noise power based on SNR = 10*log10(signal_power / noise_power)
SNR_dB = 15; % Desired SNR in dB
noise_power = signal_power / (10^(SNR_dB/10));

noise = sqrt(noise_power) * randn(size(image));

noisy_image = image + noise;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure;
imshow(image), title('Original Image');
figure;
imshow(noisy_image), title('Noisy Image (SNR 15dB)');


imwrite(noisy_image, '../images/noisy_board_15db.png');


%%%%%%
window_size = 4; % Adjust the window size as needed
smoothed_board = movmean(noisy_image, [window_size window_size]);

figure;
imshow(smoothed_board)
title('Smoothed Image using Moving Average Filter')

%%%%%%
window_size = 4;
filtered_board = medfilt2(noisy_image, [window_size window_size]);

figure;
imshow(filtered_board)
title('Filtered Image Median Filter')
