clear all;
close all;

% Read the original image
image = imread('../images/new_york.png');

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

% Generate white Gaussian noise with appropriate power based on SNR
SNR_dB = 7; % Desired SNR in dB
noise_power = var(double(degraded_image(:))); % Calculate noise power
desired_noise_power = noise_power / (10^(SNR_dB/10)); % Calculate desired noise power

% Add noise to the degraded image
noisy_image = double(degraded_image) + sqrt(desired_noise_power) * randn(size(degraded_image));

% Clip the values to ensure they are within the valid range [0, 255]
noisy_image = uint8(max(min(noisy_image, 255), 0));

% Display the noisy image
figure;
imshow(noisy_image);
title('Noisy Image with SNR 7 dB');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Wiener deconvolution
restored_image = deconvwnr(noisy_image, gaussian_kernel, 1/desired_noise_power);

% Clip the values to ensure they are within the valid range [0, 255]
restored_image = uint8(max(min(restored_image, 255), 0));

% Display the restored image
figure;
imshow(restored_image);
title('Restored Image using Wiener Deconvolution');
