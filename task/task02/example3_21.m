% Example Matlab script as provided with textbook:
%
%  Fundamentals of Digital Image Processing: A Practical Approach with Examples in Matlab
%  Chris J. Solomon and Toby P. Breckon, Wiley-Blackwell, 2010
%  ISBN: 0470844736, DOI:10.1002/9780470689776, http://www.fundipbook.com
%
I = imread('pout.tif');  % read in image
I1 = adapthisteq(I,'clipLimit',0.02,'Distribution','rayleigh');
I2 = adapthisteq(I,'clipLimit',0.02,'Distribution','exponential');
I3 = adapthisteq(I,'clipLimit',0.08,'Distribution','uniform');

subplot(2,4,1), imshow(I); title('input');
subplot(2,4,2); imhist(I);
subplot(2,4,3), imshow(I1); title('rayleigh'); % diplay orig. + output
subplot(2,4,4); imhist(I1);
subplot(2,4,5), imshow(I2); title('exponential');
subplot(2,4,6); imhist(I2);
subplot(2,4,7), imshow(I3); title('uniform'); % diplay outputs
subplot(2,4,8); imhist(I3);
