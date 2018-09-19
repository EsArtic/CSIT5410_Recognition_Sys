function n=mysiftalignment(I1,I2)
%   The function aligns two images by using the SIFT features. 
%   n=mysiftalignment(I1,I2) takes two images as inputs and returns the 
%   number of matched paris.
%
%   Algorithm:
%   Step 1.The function first detects the SIFT features in I1 and I2.
%
%   Step 2.Then it use match(I1,I2) function to find the matched pairs between 
%   the two images.
%
%   Step 3.The matched pairs returned by Step 2 are potential matches based 
%   on similarity of local appearance, many of which may be incorrect.
%   Therefore, we apply screenmatches.m to screen out incorrect matches. 
%
% - The implementation of SIFT feature extraction (sift.m) and matched 
%   keypoints extraction (match.m) are given to you. Some modifications are
%   made based on codes available at http://www.cs.ubc.ca/~lowe/keypoints/
%   and the brief description can be found in CSIT5410 lecture notes.
%   You must use the codes in the assignment package.
%  -The implementation of screening out incorrect matches screenmatches.m)
%   is also provided to you. You don't need to understand its implemenation 
%   in order to finish this assignment. Just read the introduction part of
%   the function to get the usage.
%
