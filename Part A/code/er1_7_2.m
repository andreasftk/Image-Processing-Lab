clear all;
close all;

% Read the image
I = imread('football.jpg');

% Convert the image to double
I_double = im2double(I);

% Reshape the image to [m*n, 3]
[m, n, ~] = size(I_double);
X = reshape(I_double, m*n, 3);

% Define the values of k
k_values = [2, 3, 4];

% Iterate over each value of k
for i = 1:length(k_values)
    k = k_values(i);
    
    % Apply k-means algorithm
    [IDX, centers] = kmeans(X, k);
    
    % Initialize an empty image to store segmented regions
    segmented_images = zeros(m, n, 3, k);
    
    % Construct images for each region
    for j = 1:k
        % Create a mask for pixels belonging to region j
        mask = reshape(IDX == j, [m, n]);
        
        % Apply the mask to the original image
        segmented_images(:, :, :, j) = repmat(mask, [1, 1, 3]) .* I_double;
    end
    
    % Display the segmented images with annotations
    figure;
    for j = 1:k
        subplot(1, k, j);
        imshow(segmented_images(:, :, :, j));
        title(['Segmented Image ', num2str(j)]);
    end
    sgtitle(['Segmented Images with k = ', num2str(k)]);

     % Initialize an empty image to store segmented regions
    segmented_image = zeros(m*n, 3);

 
end
