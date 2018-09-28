% myprewittedge computes a binary edge image from the given image.
%
%   g = myprewittedge(Im,T,direction) computes the binary edge image from the
%   input image Im.
%   
% The function myprewittedge, with the format g=myprewittedge(Im,T,direction), 
% computes the binary edge image from the input image Im. This function takes 
% an intensity image Im as its input, and returns a binary image g of the 
% same size as Im (mxn), with 1's where the function finds edges in Im and 0's 
% elsewhere. This function finds edges using the Prewitt approximation to the 
% derivatives with the assumption that input image values outside the bounds 
% are zero and all calculations are done using double-precision floating 
% point. The function returns g with size mxn. The image g contains edges at 
% those points where the absolute filter response is above or equal to the 
% threshold T.
%   
%       Input parameters:
%       Im = An intensity gray scale image.
%       T = Threshold for generating the binary output image. If you do not
%       specify T, or if T is empty ([ ]), myprewittedge(Im,[],direction) 
%       chooses the value automatically according to the Algorithm 1 (refer
%       to the assignment descripton).
%       direction = A string for specifying whether to look for
%       'horizontal' edges, 'vertical' edges, positive 45 degree 'pos45'
%       edges, negative 45 degree 'neg45' edges or 'all' edges.
%
%   For ALL submitted files in this assignment, 
%   you CANNOT use the following MATLAB functions:
%   edge, fspecial, imfilter, conv, conv2.
%
function g = myprewittedge(Im,T,direction)

% w = fspecial('gaussian', [5, 5], 1);
% Im = imfilter(Im, w, 'replicate');
% [g, t] = edge(Im, 'prewitt', 0.2, 'both');

% Initialize the binary image p
[m, n] = size(Im);
g = zeros(m, n);

% Algorithm 1: Automatically determined threshold
if isempty(T)
    % T = double(mean(Im(:)));
    T = 0.5 * (double(max(max(Im))) + double(min(min(Im))));
    previous = T;
    for i = 1 : 10
        indicate1 = (abs(Im) >= T);
        G1 = indicate1 .* Im;
        num1 = sum(indicate1(:));
        indicate2 = (abs(Im) < T);
        G2 = (Im < T) .* Im;
        num2 = sum(indicate2(:));
        m1 = sum(G1(:)) / num1;
        m2 = sum(G2(:)) / num2;
        T = 0.5 * (m1 + m2);
        if (0.05 * previous > abs(T - previous))
            break
        end
        previous = T;
    end
end

filter1 = [-1 -1 -1; 0 0 0; 1 1 1];
filter2 = [-1 0 1; -1 0 1; -1 0 1];
filter3 = [-1 -1 0; -1 0 1; 0 1 1];
filter4 = [0 1 1; -1 0 1; -1 -1 0];

if strcmpi(direction, 'all')
    for i = 2 : (m - 1)
        for j = 2 : (n - 1)
            curt_sub_region = [Im(i - 1, j - 1) Im(i - 1, j) Im(i - 1, j + 1);
                               Im(i, j - 1)     Im(i, j)     Im(i, j + 1);
                               Im(i + 1, j - 1) Im(i + 1, j) Im(i + 1, j + 1)];

            temp = sum(sum(filter1 .* curt_sub_region));
            if abs(temp) >= T
                g(i, j) = 1.0;
                continue
            end

            temp = sum(sum(filter2 .* curt_sub_region));
            if abs(temp) >= T
                g(i, j) = 1.0;
                continue
            end

            temp = sum(sum(filter3 .* curt_sub_region));
            if abs(temp) >= T
                g(i, j) = 1.0;
                continue
            end

            temp = sum(sum(filter4 .* curt_sub_region));
            if abs(temp) >= T
                g(i, j) = 1.0;
                continue
            end
        end
    end
else
    % Choose the filters
    if strcmpi(direction, 'horizontal')
        filter = filter1;
    end
    if strcmpi(direction, 'vertical')
        filter = filter2;
    end
    if strcmpi(direction, 'pos45')
        filter = filter3;
    end
    if strcmpi(direction, 'neg45')
        filter = filter4;
    end

    for i = 2 : (m - 1)
        for j = 2 : (n - 1)
            temp = sum(sum(filter .* [Im(i - 1, j - 1) Im(i - 1, j) Im(i - 1, j + 1);
                                      Im(i, j - 1)     Im(i, j)     Im(i, j + 1);
                                      Im(i + 1, j - 1) Im(i + 1, j) Im(i + 1, j + 1)]));
            if abs(temp) >= T
                g(i, j) = 1.0;
            end
        end
    end
end

g = im2uint8(g);
