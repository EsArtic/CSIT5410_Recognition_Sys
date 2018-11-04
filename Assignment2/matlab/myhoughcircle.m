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
Accumulator = zeros([h, w, 1]);
for i0 = 1:h
    for j0 = 1:w
	    if Imbinary(i0, j0) == 1
		    for i1 = 1:h
			    for j1 = 1:w
				    if abs((i1-i0)*(i1-i0)+(j1-j0)*(j1-j0) - r*r) < 60
					    Accumulator(i1, j1, 1) = Accumulator(i1, j1, 1) + 1;
                    end
                end
            end
        end
    end
end


% Finding local maxima in Accumulator
max_value = Accumulator(1, 1, 1);
max_x = 1;
max_y = 1;
x0detect = [];
y0detect = [];
for i = 1:h
    for j = 1:w
	    if Accumulator(i, j, 1) >= thresh
		    y0detect = [y0detect i];
			x0detect = [x0detect j];
        end
        if Accumulator(i, j, 1) > max_value
            max_y = i;
            max_x = j;
            max_value = Accumulator(i, j, 1);
        end
    end
end
max(Accumulator(:))
if length(x0detect) == 0
    y0detect = [y0detect max_y];
	x0detect = [x0detect max_x];
end
% y0detect = [max_y];
% x0detect = [max_x];
