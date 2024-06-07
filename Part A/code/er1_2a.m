clear all;
close all;

% Read and convert the image to grayscale
input_signal = rgb2gray(imread('../images/lenna.jpg'));

% Get the original dimensions
[N, M] = size(input_signal);

% Pad the image to make its dimensions multiples of 32
padded_signal = padarray(input_signal, [mod(-N, 32), mod(-M, 32)], 0, 'post');

% Convert the image to double precision
padded_signal = im2double(padded_signal);

% DCT transformation matrix
T = dctmtx(32);
dct = @(block_struct) double(T) * double(block_struct.data) * double(T');
invdct = @(block_struct) T' * block_struct.data * T;

% Predefined percentages that correspond to full diagonals
p_values = [55/1024*100, 66/1024*100, 78/1024*100, 91/1024*100, 105/1024*100, 120/1024*100, 136/1024*100, 153/1024*100, 171/1024*100, 190/1024*100, 210/1024*100, 231/1024*100, 253/1024*100, 276/1024*100, 300/1024*100, 325/1024*100, 351/1024*100, 378/1024*100, 406/1024*100, 435/1024*100, 465/1024*100, 496/1024*100];

mse_values = zeros(size(p_values));

for i = 1:length(p_values)
    p = p_values(i) / 100; % Convert percentage to fraction
   
    num_coefficients = round(p * 32^2); % Number of coefficients to keep
    matrix = zeros(32);
    
    % Fill the upper triangular part of the matrix based on the number of coefficients
    count = 0;
    for diag = 1:(2*32-1)
        for row = max(1, diag-32+1):min(diag, 32)
            col = diag - row + 1;
            if col <= 32
                matrix(row, col) = 1;
                count = count + 1;
                if count >= num_coefficients
                    break;
                end
            end
        end
        if count >= num_coefficients
            break;
        end
    end

    mask = matrix;
    B = blockproc(padded_signal, [32 32], dct);
    B2 = blockproc(B, [32 32], @(block_struct) mask .* block_struct.data);
    I2 = blockproc(B2, [32 32], invdct);
    
    % Crop the image back to the original dimensions
    I2_cropped = I2(1:N, 1:M);
    
    % Convert input_signal to double for MSE calculation
    input_signal_double = im2double(input_signal);
    
    % Calculate MSE
    mse_values(i) = mean((input_signal_double(:) - I2_cropped(:)).^2);
    
    % Display the image
    imshow(I2_cropped);
    
    % Save the image with a unique filename
    imwrite(I2_cropped, ['../images/zonal/test' num2str(i) '.jpg']);
end

% Plot the curve
figure;
plot(p_values, mse_values, 'bo-');
xlabel('Percentage of DCT Coefficients Kept (%)');
ylabel('Mean Square Error (MSE)');
title('MSE vs. Percentage of DCT Coefficients Kept');
grid on;
