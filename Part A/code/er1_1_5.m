clear all;
close all;

% Read the image and convert to grayscale
original_image = imread('../images/leaf.jpg');
leaf_image = rgb2gray(imread('../images/leaf.jpg'));


% Rotate the image by 60 degrees
rotated_leaf_image = imrotate(leaf_image, 60);
rotated_original_image = imrotate(original_image, 60);

% Slide the rotated image by (100, 60) pixels
slid_leaf_image = circshift(rotated_leaf_image, [100, 60]);
slid_original_image = circshift(rotated_original_image, [100, 60]);

% Define threshold and create binary image
threshold = 220;
binary_slid_leaf_image = slid_leaf_image >= threshold;

% Find the boundary of the leaf in the binary image
row = 336;
col = 608;
boundaries = bwtraceboundary(binary_slid_leaf_image, [col row], 'N');

% Plot the boundary on the binary image
figure;
imshow(slid_original_image);
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
cutoff_frequencies = zeros(length(percentages), 2); % Store cutoff frequencies

% Define a colormap
colormap_lines = lines(length(percentages));
for i = 1:length(percentages)
    x = percentages(i) / 100;
    num_coeffs = round(x * length(complex_descriptors));
   
    reconstructed_descriptors = complex_descriptors(center_index - num_coeffs/2 + 1 : center_index + num_coeffs/2);
  
    reconstructed_contour = ifft(ifftshift(reconstructed_descriptors));
    
    % Scale the reconstructed contour to match the image dimensions
    scaling = numel(complex_descriptors)/num_coeffs;
    reconstructed_contour = reconstructed_contour / scaling;
 
    % Plot reconstructed contour
    figure
    imshow(slid_original_image);
    hold on;
    plot(real(reconstructed_contour), imag(reconstructed_contour), 'r', 'LineWidth', 2);
    axis equal;
    title(sprintf('Reconstructed Contour using %d%% of coefficients', percentages(i)));

   % Calculate cutoff frequencies
    cutoff_frequencies(i, 1) = center_index + num_coeffs/2;
    cutoff_frequencies(i, 2) = center_index - num_coeffs/2;
end


% Plot the magnitude of FFT of complex descriptors
figure;
plot(abs(complex_descriptors));
hold on;

% Plot cutoff lines
for i = 1:size(cutoff_frequencies, 1)
    for j = 1:2
        cutoff_frequency = cutoff_frequencies(i, j);
        ylim = get(gca,'YLim');
        if j == 1
            line([cutoff_frequency cutoff_frequency], ylim, 'Color', colormap_lines(i,:), 'LineStyle', '--');
        else
            line([cutoff_frequency cutoff_frequency], ylim, 'Color', colormap_lines(i,:), 'LineStyle', '--');
        end
    end
end

title('Magnitude of FFT of Complex Descriptors');
xlabel('Frequency');
ylabel('Magnitude');