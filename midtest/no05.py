# midtest no 5
# name  : Avima Haamesha
# nim   : 10219084

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def create_gaussian_filter(ksize:int, sigma:int):
    """ replace matlab function fspecial('gaussian', ksize, sigma)\\

        ksize   : int   kernel size\\
        sigma   : int   level of gaussian blurring\\
    """
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    e = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return e / e.sum()


# for motion filter, i run it from matlab to get this filter matrix
# its "motion_X_Y", X is the length, Y is the angle of motion filter
motion_filter_len21_angle11 = np.array([
    [0, 0,  0,  0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0.00860514538292460,	0.0176466801499984,	0.0266882149170722,	0.0357297496841460,	0.0376506779583915],
    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0.000870600776966501,	0.00991213554404031,	0.0189536703111141,	0.0279952050781879,	0.0370367398452617,	0.0460782746123356,	0.0396507201674932,	0.0306091854004193,	0.0215676506333455,	0.0125261158662717,	0.00348329892415420],
    [0,	0,	0,	0,	0,	0.00217759093808220,	0.0112191257051560,	0.0202606604722298,	0.0293021952393036,	0.0383437300063774,	0.0473852647734513,	0.0383437300063774,	0.0293021952393036,	0.0202606604722298,	0.0112191257051560,	0.00217759093808220,	0,	0,	0,	0,	0],
    [0.00348329892415420,	0.0125261158662717,	0.0215676506333455,	0.0306091854004193,	0.0396507201674932,	0.0460782746123356,	0.0370367398452617,	0.0279952050781879,	0.0189536703111141,	0.00991213554404031,	0.000870600776966501,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
    [0.0376506779583915,	0.0357297496841460,	0.0266882149170722,	0.0176466801499984,	0.00860514538292460,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
])

motion_filter_len31_angle21 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00184678592219828, 0.0121025102507645, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00883561547805126, 0.0203856995295448, 0.0319357835810383, 0.0209734907791137, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00427436098241077, 0.0158244450339043, 0.0273745290853977, 0.0255347452747542, 0.0139846612232607, 0.00243457717176723, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0112631905382638, 0.0228132745897572, 0.0300959997703947, 0.0185459157189012, 0.00699583166740774, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00670193604262326, 0.0182520200941167, 0.0298021041456102, 0.0231071702145417, 0.0115570861630482, 7.00211155474779e-06, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00214068154698276, 0.0136907655984762, 0.0252408496499697, 0.0276684247101822, 0.0161183406586887, 0.00456825660719525, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00912951110283575, 0.0206795951543292, 0.0322296792058227, 0.0206795951543292, 0.00912951110283575, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00456825660719525, 0.0161183406586887, 0.0276684247101822, 0.0252408496499697, 0.0136907655984762, 0.00214068154698276, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 7.00211155474779e-06, 0.0115570861630482, 0.0231071702145417, 0.0298021041456102, 0.0182520200941167, 0.00670193604262326, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0.00699583166740774, 0.0185459157189012, 0.0300959997703947, 0.0228132745897572, 0.0112631905382638, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0.00243457717176723, 0.0139846612232607, 0.0255347452747542, 0.0273745290853977, 0.0158244450339043, 0.00427436098241077, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0.0209734907791137, 0.0319357835810383, 0.0203856995295448, 0.00883561547805126, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0.0121025102507645, 0.00184678592219828, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])


def wiener_filter(I, psf):
    """ deblur image using wiener filter with given psf\\

        I   : Any   blured image\\
        psf : Any   point spread function, 2D numpy array
    """
    # convert PSF and blured image to frequency domain
    psf_fft = np.fft.fft2(psf, s=I.shape)
    im_fft = np.fft.fft2(I)

    # estimate power spectrum of noise and blurred image
    noise_power = np.abs(np.fft.fft2(np.random.randn(*I.shape)))**2
    im_power = np.abs(im_fft)**2

    # estimate the wiener filter transfer function (WTF)
    wtf = np.conj(psf_fft) / (np.abs(psf_fft)**2 + noise_power / im_power)
    wtf = np.nan_to_num(wtf)

    # apply wiener filter
    im_wiener = np.fft.ifft2(im_fft * wtf).real
    return im_wiener


def get_ssim(img1, img2, k1=0.01, k2=0.03, L=255):
    """ compute Structural Similarity Index (SSIM) between two images.\\
        1 indicating a perfect match and -1 indicating a complete mismatch.\\

        img1  : Any   deblured images using wiener filter with certain psf\\
        img2  : Any   original image without blur\\
        k1, k2  : float constants used for stability control.\\
        L   : int   max value of pixel
    """
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    var1 = np.var(img1)
    var2 = np.var(img2)
    cov12 = np.cov(img1.flat, img2.flat)[0,1]
    numerator = (2*mean1*mean2 + c1) * (2*cov12 + c2)
    denominator = (mean1**2 + mean2**2 + c1) * (var1 + var2 + c2)
    return numerator / denominator


def get_mse_psnr_ssim(im, im_ori):
    """ analysis to determine which image is most similar with original image\\

        im  : Any  deblured images using wiener filter with certain psf\\
        im_ori  : Any   original image without blur\\
    """
    mse = np.mean( (im - im_ori)**2 )
    max_pixel_value = 255
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    ssim = get_ssim(im, im_ori)
    return mse, psnr, ssim


def analysis_histogram(hist1, hist2):
    """ compare 2 histogram to get CORRELATON, CHI-SQUARE, INTERSECTION, BHATTACHARYYA, EMD value\\

        hist1   : Any   tested histogram\\
        hist2   : Any   original histogram\\
    """
    correl = cv.compareHist(hist1, hist2, method=cv.HISTCMP_CORREL)
    chisq = cv.compareHist(hist1, hist2, method=cv.HISTCMP_CHISQR)
    inters = cv.compareHist(hist1, hist2, method=cv.HISTCMP_INTERSECT)
    bhatta = cv.compareHist(hist1, hist2, method=cv.HISTCMP_BHATTACHARYYA)
    return correl, chisq, inters, bhatta
    

def imshow_with_histogram(I, title:str, hist):
    fig = plt.figure(figsize=(11,5)); fig.suptitle(title)
    ax1 = fig.add_subplot(121)
    ax1.imshow(I, cmap='gray')
    ax2 = fig.add_subplot(122)
    ax2.plot(hist)
    fig.savefig('./midtest/img/no05_%s.png' %title)


def resume_analysis(im_dict:dict):
    df = pd.DataFrame(im_dict)
    df = df.transpose().drop(columns=['im', 'hist'])
    df = df.transpose()
    with open('./midtest/no05_output.txt', 'w') as f:
        df_string = df.to_string(header=True, index=True)
        f.write(df_string)
    print(df)


if __name__ == "__main__":
    # read original and blurred image
    im_ori = cv.imread('./midtest/img/motor_GP.jpg', cv.IMREAD_GRAYSCALE)
    im_blured = cv.imread('./midtest/img/motor_GP_blured.jpg', cv.IMREAD_GRAYSCALE)

    # prepare every PSF
    labels = [  'gaussian (ksize=7, sigma=11)', 
                'gaussian (ksize=13, sigma=21)',
                'motion (len=21, angle=11)',
                'motion (len=31, angle=21)'
    ]
    psfs = [
        create_gaussian_filter(ksize=7, sigma=11),      # fspecial('gaussian',7,11);
        create_gaussian_filter(ksize=13, sigma=21),     # fspecial('gaussian',13,21);
        motion_filter_len21_angle11,                    # fspecial('motion', 21, 11);
        motion_filter_len31_angle21                     # fspecial('motion', 31, 21);
    ]

    # deblur using wiener filter with every PSF
    im_dict = {}
    # create histogram of original image without blur and blurred image
    hist0 = cv.calcHist([im_ori], [0], None, [256], [0,255])
    hist0 = cv.normalize(hist0, hist0, 1, 0, cv.NORM_L1)
    # iterate per PSF to operate the wiener filter
    for i, psf in enumerate(psfs):
        # restore the blured image
        im_wiener = wiener_filter(im_blured, psf)
        # analysis
        mse, psnr, ssim = get_mse_psnr_ssim(im_wiener, im_ori)
        # analysis using histogram
        hist2 = cv.calcHist([np.float32(im_wiener)], [0], None, [256], [0,255])
        hist2 = cv.normalize(hist2, hist2, 1, 0, cv.NORM_L1)
        correl, chisq, inters, bhatta = analysis_histogram(hist0, hist2)
        
        item = {
            "%s" %labels[i]: {
                "im": im_wiener,
                "mse": round(mse,3),
                "psnr": round(psnr,3),
                "ssim": round(ssim,3),
                "hist": hist2,
                "correlation": round(correl,3),
                "chisquare": round(chisq,3),
                "intersection": round(inters,3),
                "bhattacharyya": round(bhatta,3)
            }
        }
        im_dict.update(item)

        # show / save every image
        imshow_with_histogram(im_wiener, labels[i], hist2)
    imshow_with_histogram(im_ori, 'original', hist0)
    
    # make resume using pandas dataframe
    resume_analysis(im_dict)
    
    plt.show()