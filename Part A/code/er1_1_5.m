clear all;
close all;

% Read the image and convert to grayscale
leaf_image = rgb2gray(imread('../images/leaf.jpg'));

% Rotate the image by 60 degrees
rotated_leaf_image = imrotate(leaf_image, 60);

% Slide the rotated image by (100, 60) pixels
slid_leaf_image = circshift(rotated_leaf_image, [100, 60]);

% Define threshold and create binary image
threshold = 220;
binary_slid_leaf_image = slid_leaf_image >= threshold;

% Find the boundary of the leaf in the binary image
row = 336;
col = 608;
boundaries = bwtraceboundary(binary_slid_leaf_image, [col row], 'N');

% Plot the boundary on the binary image
figure;
imshow(binary_slid_leaf_image);
hold on;
plot(boundaries(:, 2), boundaries(:, 1), 'r', 'LineWidth', 2);
title('Boundary on Binary Image after Rotation and Sliding');

complex = boundaries(:, 2) + 1i * boundaries(:, 1);
complex_descriptors = fftshift(fft(complex));

% Plot the magnitude of the FFT of complex descriptors
figure;
plot(abs(complex_descriptors));
title('Magnitude of FFT of Complex Descriptors after Rotation and Sliding');
xlabel('Frequency');
ylabel('Magnitude');

center_index = numel(complex_descriptors) / 2;

% Reconstruct the contour using specified percentages of the most significant coefficients
percentages = [100, 50, 10, 1]; % x percentages
for i = 1:length(percentages)
    x = percentages(i) / 100;
    num_coeffs = round(x * length(complex_descriptors));
   
    reconstructed_descriptors = complex_descriptors(center_index - num_coeffs/2 + 1 : center_index + num_coeffs/2);
  
    reconstructed_contour = ifft(ifftshift(reconstructed_descriptors));
    
    % Plot reconstructed contour
    figure;
    plot(real(reconstructed_contour), imag(reconstructed_contour), 'r', 'LineWidth', 2);
    axis equal;
    title(sprintf('Reconstructed Contour using %d%% of coefficients after Rotation and Sliding', percentages(i)));
end
