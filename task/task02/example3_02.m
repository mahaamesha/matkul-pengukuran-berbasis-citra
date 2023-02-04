A=imread('cola1.png');  % Read in 1st image
B=imread('cola2.png');  % Read in 2nd image

subplot(1,3,1), imshow(A); % Display 1st image 
subplot(1,3,2), imshow(B); % Display 2nd image

Output = imsubtract(A, B); % subtract images

subplot(1,3,3), imshow(Output); % Display result