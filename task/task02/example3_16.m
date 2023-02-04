% Example Matlab script as provided with textbook:
%
%  Fundamentals of Digital Image Processing: A Practical Approach with Examples in Matlab
%  Chris J. Solomon and Toby P. Breckon, Wiley-Blackwell, 2010
%  ISBN: 0470844736, DOI:10.1002/9780470689776, http://www.fundipbook.com
%
I=imread('rice.png'); % read in image
N=5;
Im=medfilt2(I,[N N]); % create median image
N2=15;
Im2=medfilt2(I,[N2 N2]); % create median image

subplot(1,3,1); imshow(I); title('input');
subplot(1,3,2); imshow(Im); title('median filter N=5');
subplot(1,3,3); imshow(Im2); title('median filter N=15');