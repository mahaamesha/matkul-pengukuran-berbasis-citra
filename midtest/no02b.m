F = fft2(H);
figure
subplot(1,2,1),mesh(H), xlabel('x');ylabel('y');zlabel('magnitude')
FF=fftshift(abs(F));
subplot(1,2,2),mesh(FF), xlabel('x');ylabel('y');zlabel('magnitude')
figure
mesh(FF), xlabel('freq_x');ylabel('freq_y');zlabel('magnitude')
figure
plot(1:256,20*log10(FF(:,128)), 'r'),xlabel('freq_x');ylabel('log power')