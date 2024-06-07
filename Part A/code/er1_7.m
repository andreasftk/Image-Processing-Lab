clear all;
close all;

coins = imread('../images/coins.png');

%%%
% a)

[counts, bins] = imhist(coins, 300); % Specify the number of bins explicitly

% Normalize the histogram to [0,1] range
counts = counts / sum(counts);

% Compute the optimal threshold using Otsu's method
otsuThreshold = graythresh(coins);

% Perform thresholding
binaryImage = imbinarize(coins, otsuThreshold);
binaryImage = imfill(binaryImage,'holes');


figure;
bar(bins, counts, 'hist');
title('Histogram of coins.png');
xlabel('Pixel Intensity');
ylabel('Frequency');
xlim([0, 255]); % Set the x-axis limit to match the intensity range
hold on;
line([otsuThreshold*255, otsuThreshold*255], [0, max(counts)], 'Color', 'r', 'LineWidth', 2); % Convert threshold to [0,255] range
legend('Histogram', 'Threshold');

% Show the image with the threshold applied
figure;
imshow(binaryImage);
title('Image with Otsu Threshold');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% b)

% Use binary thresholded image as a mask
maskedCoins = bsxfun(@times, coins, uint8(binaryImage)); % Apply mask to original image

% Create black background
background = zeros(size(coins), 'uint8');

% Invert binary thresholded image to get background mask
backgroundMask = ~binaryImage;
maskedBackground = bsxfun(@times, background, uint8(backgroundMask)); % Apply inverted mask to black background

% Combine masked coins and background
segmentedImage = maskedCoins + maskedBackground;

% Display the segmented image
figure;
imshow(segmentedImage);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c)

% Create a white background
whiteBackground = ones(size(coins), 'uint8') * 255;

% Invert binary thresholded image to get background mask
backgroundMask = ~binaryImage;
maskedWhiteBackground = bsxfun(@times, whiteBackground, uint8(backgroundMask)); % Apply inverted mask to white background

% Combine masked coins and white background
segmentedImageWithWhiteBackground = maskedCoins + maskedWhiteBackground;


% Display the segmented image with white background
figure
imshow(segmentedImageWithWhiteBackground);
title('Segmented Image with White Background');

% Label connected components (coins)
[labeledImage, numCoins] = bwlabel(binaryImage);

% Create a new RGB image to display each coin with a different color
RGB = label2rgb(labeledImage, 'hsv', 'k', 'shuffle');

% Overlay the labeled regions on top of the white background
segmentedRGB = bsxfun(@times, RGB, uint8(binaryImage)) + maskedWhiteBackground;


% Display the segmented image with each coin highlighted in a different color on white background
figure;
imshow(segmentedRGB);
title('Coins Highlighted, White Background');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate the diameter of each coin
props = regionprops(labeledImage,'Centroid', 'EquivDiameter');

figure;
imshow(segmentedRGB);
hold on;

for i = 1:numCoins
    % Retrieve diameter of the current coin
    diameter = props(i).EquivDiameter;
    
    % Draw a circle representing the coin on the image
    center = props(i).Centroid;

    viscircles(center, diameter/2, 'EdgeColor', 'k');
    plot(props(i).Centroid(1),props(i).Centroid(2),'ko');

    % % Display the diameter of the coin
    % text(center(1), center(2), sprintf('D = %0.1f', diameter), ...
    %     'Color', 'r', 'FontSize', 8, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end

title('Centroids and Diameter');
hold off;

