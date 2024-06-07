
clear all;
close all;
image = imread('../images/board.png');
image = im2double(image);

%%%%%%%%%%%%%%%%%


% Desired SNR in dB
desired_snr_db = 15;

% Convert SNR from dB to linear scale
desired_snr_linear = 10^(desired_snr_db / 10);

% Calculate the noise variance required for desired SNR
% Assuming 'variance_val' is the variance of the original image signal
variance_val = var(image(:));
desired_noise_variance = variance_val / desired_snr_linear;

% Add white Gaussian noise with the calculated variance
noisy_image = imnoise(image, 'gaussian', 0, desired_noise_variance);

% Calculate the noise added to the image
noise = noisy_image - image;

% Plot the distribution of the noise
figure;
histogram(noise(:), 'Normalization', 'pdf');
title('Distribution of the Noise');
xlabel('Noise Amplitude');
ylabel('Probability Density Function (PDF)');

% Display the original and noisy images for reference
figure;
subplot(1, 2, 1);
imshow(image, []);
title('Original Image');
subplot(1, 2, 2);
imshow(noisy_image, []);
title('Noisy Image');


%%%%%%
window_size = 4; % Adjust the window size as needed
smoothed_board = movmean(noisy_image, [window_size window_size]);

figure;
imshow(smoothed_board)
title('Filtered Image using Moving Average Filter')

%%%%%%
window_size = 4;
filtered_board = medfilt2(noisy_image, [window_size window_size]);

figure;
imshow(filtered_board)
title('Filtered Image Median Filter')