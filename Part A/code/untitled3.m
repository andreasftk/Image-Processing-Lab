% Load the image
original_image = imread('../images/new_york.png');
gray_image = double(original_image) / 255; % Convert to double and normalize

% Define parameters
std_dev = 1.75;
snr_db = 7;
signal_power = mean(gray_image(:)) ^ 2;
noise_power = signal_power / (10^(snr_db/10));

% Filter the image
gaussian_kernel = fspecial('gaussian', [5 5], std_dev);
blurred_image = imfilter(gray_image, gaussian_kernel, 'conv', 'replicate');

% Add Gaussian noise
noisy_image = imnoise(blurred_image, 'gaussian', 0, noise_power);

% Wiener deconvolution with known noise power
restored_image_known = deconvwnr(noisy_image, gaussian_kernel, noise_power);

% Estimate the noise power from the noisy image
estimated_noise_power = var(noisy_image(:) - blurred_image(:));

% Wiener deconvolution with estimated noise power
restored_image_estimated = deconvwnr(noisy_image, gaussian_kernel, estimated_noise_power);



% Original and degraded images
figure;
imshow(gray_image);
title('Original Image');

figure;
imshow(noisy_image);
title('Degraded Image');

% Impulse and frequency responses
figure;
imshow(gaussian_kernel, []);
title('Impulse Response');

figure;
frequency_response = fftshift(abs(fft2(gaussian_kernel, size(gray_image, 1), size(gray_image, 2))));
imshow(log(1 + frequency_response), []);
title('Frequency Response');

% Restored images
figure;
imshow(restored_image_known);
title('Restored Image (Known Noise Power)');

figure;
imshow(restored_image_estimated);
title('Restored Image (Estimated Noise Power)');