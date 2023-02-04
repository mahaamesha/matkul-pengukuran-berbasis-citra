A=imread('cameraman.tif'); % read in image
subplot(1,2,1), imshow(A); % display image
B = imcomplement(A); % invert the image
subplot(1,2,2), imshow(B); % display result image B