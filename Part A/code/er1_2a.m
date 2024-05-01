clear all;
close all;
input_signal=rgb2gray((imread('../images/lenna.jpg')));
[N, M] = size(input_signal);
input_signal = padarray(input_signal, [mod(N,32), mod(M,32)], 0, 'post');
input_signal = input_signal(1:256, 1:256);
input_signal= im2double(input_signal);

T = dctmtx(32);
dct = @(block_struct) double(T) * double(block_struct.data) * double(T');
invdct = @(block_struct) T' * block_struct.data * T;

p_range = 5:5:50; % Range of p values
mse_values = zeros(size(p_range));

for i = 1:length(p_range)
    p = p_range(i) / 100; % Convert percentage to fraction
    matrix = zeros(32);
    num_coefficients = round(p * 32^2); % Number of coefficients to keep
    for k = 1:num_coefficients
        [row, col] = ind2sub([32, 32], k);
        matrix(row, col) = 1;
    end

    mask = matrix;
    B = blockproc(input_signal,[32 32],dct);
    B2 = blockproc(B,[32 32],@(block_struct) mask .* block_struct.data);
    I2 = blockproc(B2,[32 32],invdct);
    
    % Calculate MSE
    mse_values(i) = mean((input_signal(:) - I2(:)).^2);
end

% Plot the curve
figure;
plot(p_range, mse_values, 'bo-');
xlabel('Percentage of DCT Coefficients Kept (%)');
ylabel('Mean Square Error (MSE)');
title('MSE vs. Percentage of DCT Coefficients Kept');
grid on;
