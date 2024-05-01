close all;
clear all;

% Read the image
image = imread('../images/hallway.png');


% Apply Sobel masks to estimate gradient components
sobel_x = [-1 -2 -1; 0 0 0; 1 2 1];
sobel_y = [-1 0 1; -2 0 2; -1 0 1];

gradient_x = conv2(double(image), sobel_x, 'same');
gradient_y = conv2(double(image), sobel_y, 'same');

gradient = [gradient_x; gradient_y];

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
imshow(abs_gradient_x_gamma);
title('Absolute Gradient X');

figure;
imshow(abs_gradient_y_gamma);
title('Absolute Gradient Y');

figure;
imshow(gradient_amplitude_gamma);
title('Gradient Amplitude');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set the threshold value (adjust as needed)
threshold = 30;

% Threshold the gradient amplitude image
thresholded_image = gradient_amplitude_gamma > threshold;

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%  https://www.geeksforgeeks.org/matlab-image-edge-detection-using-sobel-operator-from-scratch/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%edges = edge(image, 'Sobel');

[H, theta, rho] = hough(thresholded_image);

% Find peaks in Hough transform
peaks = houghpeaks(H, 10); % Adjust the number of peaks as needed

% Extract lines from Hough transform

% Find lines in the image
lines = houghlines(thresholded_image,theta,rho, peaks);


% Plot the detected lines on the original image
figure;
imshow(image);
hold on;

for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'red');
    % Plot beginnings and ends of lines
    plot(xy(1,1), xy(1,2), 'x', 'LineWidth', 2, 'Color', 'green');
    plot(xy(2,1), xy(2,2), 'x', 'LineWidth', 2, 'Color', 'blue');
end
hold off;
title('Detected Line Segments on Original Image');
