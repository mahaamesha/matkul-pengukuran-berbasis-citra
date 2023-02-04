% Example Matlab script as provided with textbook:
%
%  Fundamentals of Digital Image Processing: A Practical Approach with Examples in Matlab
%  Chris J. Solomon and Toby P. Breckon, Wiley-Blackwell, 2010
%  ISBN: 0470844736, DOI:10.1002/9780470689776, http://www.fundipbook.com
%
I=imread('coins.png'); % read in image
subplot(1,2,1); imshow(I); title('coins.png');
level = graythresh(I); % get OTSU theshold
It = im2bw(I, level);  % theshold image
subplot(1,2,2); imshow(It); title(level); % display it
