clear all;
close all;

% Read the original image
original_image = imread('../images/board.png');

% Specify the density of the impulse noise (30%)
density = 0.3;

% Add impulse noise to the image
noisy_image = imnoise(original_image, 'salt & pepper', density);

% Display the original and noisy images
subplot(1, 2, 1);
imshow(original_image);
title('Original Image');

subplot(1, 2, 2);
imshow(noisy_image);
title('Noisy Image (with 30% impulse noise)');

%%%%%%
window_size = 7; % Adjust the window size as needed
filtered_board = movmean(noisy_image, [window_size window_size]);

filtered_board= uint8(filtered_board);

figure;
imshow(filtered_board)
title('Filtered Image using Moving Average Filter');

%%%%%%
window_size = 6;
filtered_board = medfilt2(noisy_image, [window_size window_size]);

figure;
imshow(filtered_board)
title('Filtered Image using Median Filter')
