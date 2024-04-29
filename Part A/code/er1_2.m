close all;
clear all;
% Read the image
img = imread('lenna.jpg');
figure;
imshow(uint8(img));

% Convert to grayscale if necessary
if size(img, 3) == 3
    img = rgb2gray(img);
end

% Get the current dimensions of the image
[numRows, numCols] = size(img);

% Calculate the padding needed to make the dimensions divisible by 32
padRows = mod(32 - mod(numRows, 32), 32);
padCols = mod(32 - mod(numCols, 32), 32);

% Pad the image with zeros
paddedImg = padarray(img, [padRows, padCols], 0, 'post');

% Update the dimensions of the padded image
[numRowsPadded, numColsPadded] = size(paddedImg);

% Now, the dimensions of the padded image are divisible by 32

% Slice into non-overlapping regions
numRegionsRows = numRowsPadded / 32;
numRegionsCols = numColsPadded / 32;

% Convert to integers to avoid the error
numRegionsRows = round(numRegionsRows);
numRegionsCols = round(numRegionsCols);

% Initialize cell array to store the regions
regions = cell(numRegionsRows, numRegionsCols);

% Loop through the image and extract regions
for i = 1:numRegionsRows
    for j = 1:numRegionsCols
        region = paddedImg((i-1)*32+1 : i*32, (j-1)*32+1 : j*32);
        regions{i, j} = region;
    end
end

% Apply 2D-DCT transform to each region
dctRegions = cell(numRegionsRows, numRegionsCols);

for i = 1:numRegionsRows
    for j = 1:numRegionsCols
        dctRegions{i, j} = dct2(regions{i, j});
    end
end

figure;
% Plot the original image
subplot(2, 2, 1);
imshow(img);
title('Original Image');

% Plot the padded image
subplot(2, 2, 2);
imshow(paddedImg);
title('Padded Image');

% Plot the regions
subplot(2, 2, 3);
hold on;
for i = 1:numRegionsRows
    for j = 1:numRegionsCols
        x = (j-1)*32+1;
        y = (i-1)*32+1;
        rectangle('Position', [x, y, 32, 32], 'EdgeColor', 'r');
    end
end
hold off;
title('Regions');

% Plot the result (DCT of regions)
subplot(2, 2, 4);
imshow(cell2mat(dctRegions), []);
title('DCT of Regions');

% Define the p-value for coefficient selection (adjust as needed)
p = 0.1; % Example: retain the top 10% of coefficients

% Select the subset of DCT coefficients for compression
compressedDCTRegions = cell(numRegionsRows, numRegionsCols);
for i = 1:numRegionsRows
    for j = 1:numRegionsCols
        % Get the size of the current DCT region
        [m, n] = size(dctRegions{i, j});
        
        % Calculate the number of coefficients to keep
        numCoefficients = round(p * m * n);
        
        % Sort the absolute values of DCT coefficients in descending order
        [~, sortedIndices] = sort(abs(dctRegions{i, j}(:)), 'descend');
        
        % Set all but the top 'numCoefficients' coefficients to zero
        compressedDCT = zeros(m, n);
        compressedDCT(sortedIndices(1:numCoefficients)) = dctRegions{i, j}(sortedIndices(1:numCoefficients));
        
        compressedDCTRegions{i, j} = compressedDCT;
    end
end

% Plot the result (Compressed DCT of regions)
figure;
subplot(1, 2, 1);
imshow(cell2mat(dctRegions), []);
title('Original DCT of Regions');

subplot(1, 2, 2);
imshow(cell2mat(compressedDCTRegions), []);
title('Compressed DCT of Regions');


% Inverse transform the compressed DCT regions
reconstructedRegions = cell(numRegionsRows, numRegionsCols);
for i = 1:numRegionsRows
    for j = 1:numRegionsCols
        % Inverse DCT transform
        reconstructedRegions{i, j} = idct2(compressedDCTRegions{i, j});
    end
end

% Reconstruct the padded image from the regions
reconstructedPaddedImg = zeros(numRowsPadded, numColsPadded);
for i = 1:numRegionsRows
    for j = 1:numRegionsCols
        x = (j-1)*32+1;
        y = (i-1)*32+1;
        reconstructedPaddedImg(y:y+31, x:x+31) = reconstructedRegions{i, j};
    end
end

% Crop the reconstructed padded image to the original size
reconstructedImg = reconstructedPaddedImg(1:numRows, 1:numCols);

% Display the compressed image
figure;
imshow(uint8(reconstructedImg));
title('Compressed Image');