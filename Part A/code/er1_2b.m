clear all;
close all;

input_signal = rgb2gray(imread('../images/lenna.jpg'));

[N, M] = size(input_signal);

input_signal = padarray(input_signal, [mod(N,32), mod(M,32)], 0, 'post');
input_signal = input_signal(1:256, 1:256);
input_signal = im2double(input_signal);

T = dctmtx(32);
dct = @(block_struct) double(T) * double(block_struct.data) * double(T');
invdct = @(block_struct) T' * block_struct.data * T;

threshold_range = 0.05:0.05:0.5;  % Range of thresholds from 5% to 50%
mse_values = zeros(size(threshold_range));

for i = 1:length(threshold_range)
    threshold = threshold_range(i);
    B = blockproc(input_signal, [32 32], dct);
    B2 = blockproc(B, [32 32], @(block_struct) threshold_mask(block_struct.data, threshold));
    I2 = blockproc(B2, [32 32], invdct);
    mse_values(i) = immse(input_signal, I2);
end

figure;
plot(threshold_range, mse_values, 'bo-');
xlabel('Threshold');
ylabel('Mean Square Error (MSE)');
title('MSE vs. Threshold');
grid on;

function masked_block = threshold_mask(block, threshold)
    a = (abs(block) > threshold);
    masked_block = block .* a;
end

