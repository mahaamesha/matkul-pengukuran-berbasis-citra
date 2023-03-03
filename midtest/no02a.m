clear all
close all

PSF = 30;
f = ones(256,256);
u = size(f)/2;
[u,v] = ndgrid(-u(1):u(1)-1,-u(2):u(2)-1);
H = u.^2+v.^2 < PSF.^2; 
figure
imshow(H)
disp(u)

figure
mesh(u,v,zeros(size(u)))
xlabel('u')
ylabel('v')
zlabel('Magnitude')
title('Frequency grid')