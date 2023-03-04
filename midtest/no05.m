PSF1 = fspecial('gaussian', 7, 11);
disp('PSF1'), disp(PSF1);

PSF2 = fspecial('motion', 5, 3);
disp('PSF2'), disp(PSF2);

PSF1 = fspecial('gaussian',7,11); disp('PSF1'), disp(PSF1);
PSF2 = fspecial('motion', 21, 11); disp('PSF2'), disp(PSF2);
PSF3 = fspecial('gaussian',13,21); disp('PSF3'), disp(PSF3);
PSF4 = fspecial('motion', 31, 21); disp('PSF4'), disp(PSF4);
