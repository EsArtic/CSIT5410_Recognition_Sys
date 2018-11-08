function [y0detect,x0detect,Accumulator] = houghcircle(Imbinary,r,thresh)
%HOUGHCIRCLE - detects circles with specific radius in a binary image.
%
%Comments:
%       Function uses Standard Hough Transform to detect circles in a binary image.
%       According to the Hough Transform for circles, each pixel in image space
%       corresponds to a circle in Hough space and vice versa. 
%       upper left corner of image is the origin of coordinate system.
%
%Usage: [y0detect,x0detect,Accumulator] = houghcircle(Imbinary,r,thresh)
%
%Arguments:
%       Imbinary - a binary image. image pixels that have value equal to 1 are
%                  interested pixels for HOUGHLINE function.
%       r        - radius of circles.
%       thresh   - a threshold value that determines the minimum number of
%                  pixels that belong to a circle in image space. threshold must be
%                  bigger than or equal to 4(default).
%
%Returns:
%       y0detect    - row coordinates of detected circles.
%       x0detect    - column coordinates of detected circles. 
%       Accumulator - the accumulator array in Hough space.

if nargin == 2
    thresh = 4;
elseif thresh < 4
    error('treshold value must be bigger or equal to 4');
    return
end

%Voting
[h, w] = size(Imbinary);
Accumulator = zeros([h, w]);
for y0 = 1:h
    for x0 = 1:w
        if Imbinary(y0, x0) == 1
            boundaries = [];
            for angle = 0:pi/180:2*pi
                y = floor(y0 + r * sin(angle));
                x = floor(x0 + r * cos(angle));
                if (0 < y) && (y <= h)
                    if (0 < x) && (x <= w)
                        boundaries = [boundaries; [y, x]];
                    end
                end
            end
            boundaries = unique(boundaries, 'rows');
            [n, m] = size(boundaries);
            for i = 1:n
                y1 = boundaries(i, 1);
                x1 = boundaries(i, 2);
                Accumulator(y1, x1) = Accumulator(y1, x1) + 1;
            end
        end
    end
end


% Finding local maxima in Accumulator
max_value = Accumulator(1, 1);
max_x = 1;
max_y = 1;
x0detect = [];
y0detect = [];
for i = 1:h
    for j = 1:w
        if Accumulator(i, j) >= thresh
            y0detect = [y0detect i];
            x0detect = [x0detect j];
        end
        if Accumulator(i, j) > max_value
            max_y = i;
            max_x = j;
            max_value = Accumulator(i, j);
        end
    end
end
if length(x0detect) == 0
    y0detect = [y0detect max_y];
    x0detect = [x0detect max_x];
end
