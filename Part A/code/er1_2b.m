clear all;
close all;

% Load and preprocess the image
input_signal = rgb2gray(imread('../images/lenna.jpg'));

[N, M] = size(input_signal);

padded_signal = padarray(input_signal, [mod(-N, 32), mod(-M, 32)], 0, 'post');
padded_signal = im2double(padded_signal);

% Define DCT and inverse DCT functions
T = dctmtx(32);
dct = @(block_struct) double(T) * double(block_struct.data) * double(T');
invdct = @(block_struct) T' * block_struct.data * T;

% Define the percentage range and MSE values array
percentages = 5:1:50;
mse_values = zeros(size(percentages));

for p_idx = 1:length(percentages)
    p = percentages(p_idx) / 100;

    % Apply DCT to each block
    B = blockproc(padded_signal, [32 32], dct);

    % Retain top p% coefficients (threshold method)
    B2 = blockproc(B, [32 32], @(block_struct) retain_top_percentage(block_struct.data, p));

    % Apply inverse DCT to each block
    I2 = blockproc(B2, [32 32], invdct);

    I2_cropped = I2(1:N, 1:M);
    
    % Convert input_signal to double for MSE calculation
    input_signal_double = im2double(input_signal);

    % Compute and store MSE
    mse_values(p_idx) = mean((input_signal_double(:) - I2_cropped(:)).^2);

    % Save the compressed image (optional)
    imwrite(I2_cropped, sprintf('../images/thres_comp/lenna_compressed_p%d.jpg', round(p * 100)), 'jpg');
end

% Plot the MSE curve
figure;
plot(percentages, mse_values, '-o');
xlabel('Percentage of DCT Coefficients Retained');
ylabel('Mean Squared Error (MSE)');
title('MSE between Original and Reconstructed Image');
grid on;

function masked_block = retain_top_percentage(block, p)
    % Sort DCT coefficients by magnitude
    dct_sorted = sort(abs(block(:)), 'descend');
    
    % Determine the threshold value for top p% coefficients
    threshold = dct_sorted(round(p * numel(dct_sorted)));
    
    % Zero out coefficients below the threshold
    masked_block = block;
    masked_block(abs(block) < threshold) = 0;
end
