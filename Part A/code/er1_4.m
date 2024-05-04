close all;
clear all;

% Read the images
dark_road_1 = imread('../images/dark_road_1.jpg');
dark_road_2 = imread('../images/dark_road_2.jpg');
dark_road_3 = imread('../images/dark_road_3.jpg');

figure;
imshow(dark_road_1);
figure;
imshow(dark_road_2);
figure;
imshow(dark_road_3);

figure;
imhist(dark_road_1);
title('Histogram of dark\_road\_1.jpg');
figure;
imhist(dark_road_2);
title('Histogram of dark\_road\_2.jpg');
figure;
imhist(dark_road_3);
title('Histogram of dark\_road\_3.jpg');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eq_dark_road_1 = histeq(dark_road_1);
eq_dark_road_2 = histeq(dark_road_2);
eq_dark_road_3 = histeq(dark_road_3);

figure;
imhist(eq_dark_road_1);
title('Histogram of dark\_road\_1.jpg');

figure;
imhist(eq_dark_road_2);
title('Histogram of dark\_road\_2.jpg');

figure;
imhist(eq_dark_road_3);
title('Histogram of dark\_road\_3.jpg');

figure;
imshow(eq_dark_road_1);
figure;
imshow(eq_dark_road_2);
figure;
imshow(eq_dark_road_3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define window sizes to test
window_size = 2;

% Apply adaptive histogram equalization with best window size
eq_dark_road_1 = adapthisteq(dark_road_1, 'NumTiles', [window_size, window_size]);
eq_dark_road_2 = adapthisteq(dark_road_2, 'NumTiles', [window_size, window_size]);
eq_dark_road_3 = adapthisteq(dark_road_3, 'NumTiles', [window_size, window_size]);


figure;
imhist(eq_dark_road_1);
title('Histogram of dark\_road\_1.jpg');

figure;
imhist(eq_dark_road_2);
title('Histogram of dark\_road\_2.jpg');

figure;
imhist(eq_dark_road_3);
title('Histogram of dark\_road\_3.jpg');


figure;
imshow(eq_dark_road_1);
figure;
imshow(eq_dark_road_2);
figure;
imshow(eq_dark_road_3);