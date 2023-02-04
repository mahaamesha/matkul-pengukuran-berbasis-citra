A=imread('cameraman.tif'); % Read in image

subplot(1,2,1), imshow(A); % Display image

B = imadd(A, 100); % Add 100 pixel values to image A

subplot(1,2,2), imshow(B); % Display result image B