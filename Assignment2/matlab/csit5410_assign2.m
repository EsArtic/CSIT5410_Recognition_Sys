% CSIT5410M_ASSIGN2.m runs CSIT5410 assignment 2 functions.
% 
%   CSIT5410_ASSIGN2 performs Hough Circle Detection

%   REMINDER: You cannot modify this file.

close all
clear all
clc

% TASK 1
disp('*********************************************');
disp('Task 1: Hough Circle Detection');
disp('*********************************************');
I = im2double(imread('qiqiu.png'));
I = mean(I, 3);
figure; imshow(I);
E=edge(I,'canny');
figure;imshow(E,[]);
radius = 114;
thres = 85;
[y0detect,x0detect,Accumulator] = myhoughcircle(E,radius,thres);

[V,co] = max(max(Accumulator(:,:)));

figure;
imagesc(Accumulator);title('Accmulator cells');

mask = zeros(512,512);
for k = 1:length(x0detect)
    Xc = y0detect(k); % y0detect indicates the row number
    Yc = x0detect(k); % x0detect indicates the column number
    mask = draw_circle(mask, Xc, Yc, radius);
end
figure,imshow(mask, []);

% TASK 2
disp('*********************************************');
disp('Task 2: Fisher Linear Discriminant');
disp('*********************************************');
class1_samples=[1 2;2 3;3 3;4 5;5 5]; % each row represents a pair of x and y coordinates
class2_samples=[1 0;2 1;3 1;3 2;5 3;6 5]; % each row represents a pair of x and y coordinates
input_sample=[2 5]; % [x y] coordinates
[output_class, w, s_w, mean_c1, mean_c2] = myfld(input_sample, class1_samples, class2_samples);
fprintf('Output class is %d.\n\n', output_class)
fprintf('Within-class scatter matrix:\n')
disp(s_w)
fprintf('Weights:\n')
disp(w)
