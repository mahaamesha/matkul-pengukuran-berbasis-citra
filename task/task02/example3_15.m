% Example Matlab script as provided with textbook:
%
%  Fundamentals of Digital Image Processing: A Practical Approach with Examples in Matlab
%  Chris J. Solomon and Toby P. Breckon, Wiley-Blackwell, 2010
%  ISBN: 0470844736, DOI:10.1002/9780470689776, http://www.fundipbook.com
%
I=imread('rice.png'); % read in image
Im=imfilter(I,fspecial('average',[15 15]),'replicate'); % create mean image
It = I - (Im + 20); % subtract mean image (+ constant C=20)
Ibw=im2bw(It,0);    % threshold result at 0 (keep +ve results only)
subplot(2,3,1), imshow(I); title('input'); % Display image
subplot(2,3,2), imshow(Im); title('average filter');
subplot(2,3,3), imshow(Im+20); title('(Im + 20)');
subplot(2,3,4), imshow(It); title('I - (Im + 20)');
subplot(2,3,5), imshow(Ibw); title('output'); % Display result

