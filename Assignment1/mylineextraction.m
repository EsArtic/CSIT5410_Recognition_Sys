function [bp, ep] = mylineextraction(BW)
%   The function extracts the longest line segment from the given binary image
%       Input parameter:
%       BW = A binary image.
%
%       Output parameters:
%       [bp, ep] = beginning and end points of the longest line found
%       in the image.
%
%   You may need the following predefined MATLAB functions: hough,
%   houghpeaks, houghlines.
[H, T, R] = hough(BW, 'Theta',-90:0.5:89);

% Peaks = houghpeaks(H, 5);
Peaks = houghpeaks(H, 5, 'threshold', ceil(0.2 * max(H(:))))

lines = houghlines(BW, T, R, Peaks);

[m, n] = size(BW);
index = 1;
maxLen = 0;
for i = 1:length(lines)
    x = lines(i).point1(1) - lines(i).point2(1);
    y = lines(i).point1(2) - lines(i).point2(2);
    currLen = x * x + y * y;
    if currLen > maxLen
        index = i;
        maxLen = currLen;
    end
end

bp = lines(index).point1;
ep = lines(index).point2;
