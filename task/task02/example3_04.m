A=imread('cola1.png');  % Read in 1st image
subplot(1,3,1), imshow(A); % Display 1st image 

Output = immultiply(A,1.5);     % multiple image by 1.5
subplot(1,3,2), imshow(Output); % Display result

Output = imdivide(A,4);         % divide image by 4
subplot(1,3,3), imshow(Output); % Display result