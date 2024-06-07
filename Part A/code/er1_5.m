clear all;
close all;

% Read the original image
image = imread('../images/new_york.png');
image = im2double(image);

% Display the original image
figure;
imshow(image);
title('Original Image');

% Define the standard deviation for Gaussian smoothing (in the suggested range)
std_dev = 1.8; % You can adjust this value within the suggested range

% Create a 2D Gaussian smoothing kernel
kernel_size = 2 * ceil(2 * std_dev) + 1;
gaussian_kernel = fspecial('gaussian', [kernel_size, kernel_size], std_dev);

% Apply Gaussian smoothing to degrade the image
degraded_image = imfilter(image, gaussian_kernel, 'conv', 'replicate');

% Display the degraded image
figure;
imshow(degraded_image);
title('Degraded Image');

% Plot the impulse response (shock response)
figure;
subplot(1, 2, 1);
imshow(gaussian_kernel, []);
title('Impulse Response (Shock Response)');
xlabel('Spatial Domain');
ylabel('Spatial Domain');

% Compute and plot the frequency response
frequency_response = fftshift(fft2(gaussian_kernel));
magnitude_response = abs(frequency_response);
magnitude_response = log(1 + magnitude_response); % Log scale for better visualization

subplot(1, 2, 2);
imshow(magnitude_response, []);
title('Frequency Response');
xlabel('Frequency Domain');
ylabel('Frequency Domain');
colormap(gca, jet);
colorbar;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Desired SNR in dB
desired_snr_db = 7;

% Convert SNR from dB to linear scale
desired_snr_linear = 10^(desired_snr_db / 10);

% Calculate the noise variance required for desired SNR
% Assuming 'variance_val' is the variance of the original image signal
variance_val = var(image(:));
desired_noise_variance = variance_val / desired_snr_linear;

% Add white Gaussian noise with the calculated variance
noisy_image = imnoise(image, 'gaussian', 0, desired_noise_variance);

% Display the noisy image
figure;
imshow(noisy_image);
title('Noisy Image with SNR 7 dB');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Wiener deconvolution
noise_power_ratio = desired_noise_variance / var(noisy_image(:)); % noise-to-signal power ratio
restored_image = deconvwnr(noisy_image, gaussian_kernel, noise_power_ratio);

% Clip the values to ensure they are within the valid range [0, 1]
restored_image = max(min(restored_image, 1), 0);



