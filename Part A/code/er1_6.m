close all;
clear all;

% Read the image
image = imread('../images/hallway.png');

% Convert image to grayscale if it is not already
if size(image, 3) == 3
    gray_image = rgb2gray(image);
else
    gray_image = image;
end

% Apply Sobel masks to estimate gradient components
sobel_x = [-1 -2 -1; 0 0 0; 1 2 1];
sobel_y = [-1 0 1; -2 0 2; -1 0 1];

gradient_x = conv2(double(gray_image), sobel_x, 'same');
gradient_y = conv2(double(gray_image), sobel_y, 'same');

% Compute absolute values of gradient components
abs_gradient_x = abs(gradient_x);
abs_gradient_y = abs(gradient_y);

% Compute amplitude of the gradient
gradient_amplitude = sqrt(gradient_x.^2 + gradient_y.^2);

% Apply gamma correction for visualization
gamma = 0.5; % adjust as needed
abs_gradient_x_gamma = uint8(255 * (abs_gradient_x / max(abs_gradient_x(:))).^gamma);
abs_gradient_y_gamma = uint8(255 * (abs_gradient_y / max(abs_gradient_y(:))).^gamma);
gradient_amplitude_gamma = uint8(255 * (gradient_amplitude / max(gradient_amplitude(:))).^gamma);

% Display gradient images
figure;
imshow(abs_gradient_x_gamma);
title('Absolute Gradient X');

figure;
imshow(abs_gradient_y_gamma);
title('Absolute Gradient Y');

figure;
imshow(gradient_amplitude_gamma);
title('Gradient Amplitude');

% Set the threshold value (adjust as needed)
threshold = 40;

% Threshold the gradient amplitude image
thresholded_image = gradient_amplitude > threshold; % use original gradient amplitude for thresholding

% Draw histogram of gradient amplitude image
figure;
histogram(gradient_amplitude(:), 'Normalization', 'probability');
hold on;
line([threshold threshold], ylim, 'LineWidth', 2, 'Color', 'r');
hold off;
title('Histogram of Gradient Amplitude Image');
xlabel('Gradient Amplitude');
ylabel('Probability');

% Show the thresholded result
figure;
imshow(thresholded_image);
title('Thresholded Gradient Amplitude Image');

% Apply the Hough transform to find lines
[H, T, R] = hough(thresholded_image);

% Find peaks in the Hough transform matrix
P = houghpeaks(H, 20, 'threshold', ceil(0.3 * max(H(:)))); % Increase number of peaks

% Find lines corresponding to the Hough transform peaks
lines = houghlines(thresholded_image, T, R, P, 'FillGap', 5, 'MinLength', 40); % Adjust FillGap and MinLength

% Margin to ignore lines close to the border
margin = 0.1;

% Display the original image
figure;
imshow(image);
hold on;

% Plot the detected lines on the image, excluding those too close to the borders
for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    
    % Check if the line is within the margin
    if all(xy(:, 1) > margin) && all(xy(:, 1) < size(image, 2) - margin) && ...
       all(xy(:, 2) > margin) && all(xy(:, 2) < size(image, 1) - margin)
       
        plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'g'); % Change color as needed

        % Plot the beginnings and ends of lines
        plot(xy(1,1), xy(1,2), 'x', 'LineWidth', 2, 'Color', 'yellow');
        plot(xy(2,1), xy(2,2), 'x', 'LineWidth', 2, 'Color', 'red');
    end
end

title('Detected Line Segments on Original Image');
hold off;

