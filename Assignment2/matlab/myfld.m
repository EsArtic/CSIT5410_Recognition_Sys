% MYFLD classifies an input sample into either class 1 or class 2.
%
%   [output_class, w, s_w, mean_c1, mean_c2] = myfld(input_sample, class1_samples, class2_samples)
%   classifies an input sample into either class 1 or class 2,
%   from samples of class 1 (class1_samples) and samples of
%   class 2 (class2_samples).
% 
% The implementation of the Fisher linear discriminant must follow the
% descriptions given in the lecture notes.
% In this assignment, you do not need to handle cases when 'inv' function
% input is a matrix which is badly scaled, singular or nearly singular.
% All calculations are done using double-precision floating point. 
%
% Input parameters:
% input_sample = an input sample
%   - The number of dimensions of the input sample is N.
%
% class1_samples = a NC1xN matrix
%   - class1_samples contains all samples taken from class 1.
%   - The number of samples is NC1.
%   - The number of dimensions of each sample is N.
%
% class2_samples = a NC2xN matrix
%   - class2_samples contains all samples taken from class 2.
%   - The number of samples is NC2.
%   - NC1 and NC2 do not need to be the same.
%   - The number of dimensions of each sample is N.
%
% Output parameters:
% output_class = the class to which input_sample belongs. 
%   - output_class should have the value either 1 or 2.
%
% w = weight vector.
%   - The vector length must be one.
%
% s_w = within-class scatter matrix
%
% mean_c1 = mean vector of Class 1 samples
%
% mean_c2 = mean vector of Class 2 samples
%
% For ALL submitted files in this assignment, 
%   you CANNOT use the following MATLAB functions:
%   mean, diff, classify, classregtree, eval, mahal.
function [output_class, w, s_w, mean_c1, mean_c2] = myfld(input_sample, class1_samples, class2_samples)

[n1, n] = size(class1_samples);
[n2, n] = size(class2_samples);

mean_c1 = sum(class1_samples) / n1;
fprintf('Mean of class 1:\n')
disp(mean_c1)

mean_c2 = sum(class2_samples) / n2;
fprintf('Mean of class 2:\n')
disp(mean_c2)

s1 = 0;
for i = 1:n1
    qi = class1_samples(i, :);
    s1 = s1 + (qi - mean_c1)' * (qi - mean_c1);
end
fprintf('Within class variance of class 1:\n')
disp(s1)

s2 = 0;
for i = 1:n2
    qi = class2_samples(i, :);
    s2 = s2 + (qi - mean_c2)' * (qi - mean_c2);
end
fprintf('Within class variance of class 2:\n')
disp(s2)

s_w = s1 + s2;
fprintf('Total within class variance:\n')
disp(s_w)

s_b = (mean_c2 - mean_c1)' * (mean_c2 - mean_c1);
fprintf('Between class variance:\n')
disp(s_b)

w = inv(s_w) * (mean_c2 - mean_c1)';
fprintf('Weight:\n')
disp(w)

seperation_point = 0.5 * w' * (mean_c1 + mean_c2)';
fprintf('Seperation point:\n')
disp(seperation_point)

value = input_sample * w;
fprintf('The product of weight and input sample:\n')
disp(value)

if value < seperation_point
    output_class = 1;
else
    output_class = 2;
end
