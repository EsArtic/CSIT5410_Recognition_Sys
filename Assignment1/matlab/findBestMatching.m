function n = findBestMatching(I, I1, I2, I3)
num = zeros(1,3);

[des0, loc0] = sift(I);

% mysiftalignment is used to align two images by using SIFT features 
num(1)=mysiftalignment(I, des0, loc0, I1, '05QR_img1.png');
num(2)=mysiftalignment(I, des0, loc0, I2, '06QR_img2.png');
num(3)=mysiftalignment(I, des0, loc0, I3, '07QR_img3.png');

% Find the image with the largest number of matched pairs with the QR code image. 
[tp n]=max(num);