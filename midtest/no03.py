# midtest no 3
# name  : Avima Haamesha
# nim   : 10219084

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def remove_noise(I, ksize:int, filter:str):
    """ filtering image to remove salt n pepper noise\\

        I   : Any   input image\\
        ksize   : int   kernel size\\
        filter  : str   gaussian/mean/median\\
    """
    if filter == 'gaussian': im = cv.GaussianBlur(I, (ksize,ksize), sigmaX=0)   # if i set sigmaX=0, it calculated using sigma_x = (n_x - 1)/2 * 0.3 + 0.8
    elif filter == 'mean': im = cv.boxFilter(I, -1, (ksize,ksize))      # ddepth=-1, destination im will have same depth as the source
    elif filter == 'median': im = cv.medianBlur(I, ksize)
    return im


def calculate_SNR(I,I2=None, method:bool=1):
    """ to calculate Signal to Noise Ratio (SNR)\\

        I   : Any   input of filtered image\\
        I2  : Any   input of original noisy image\\
        method  : bool  if 0, calculated with my preferred formula
                        if 1, calculated with variance
    """
    if method == 0:
        signal = np.mean(I)
        if I2 == None: noise = np.std(I)
        else: noise = np.std(I - I2)    # filtered - original noisy image
        snr = 20 * np.log10(signal / noise)
    else:
        roi_b = I[10:60, 200:250]   # background region
        roi_a = I[60:90, 100:130]   # forehead region
        # cv.imshow('roi_a', roi_a)q; cv.imshow('roi_b', roi_b); cv.waitKey(0)
        if np.var(roi_b) != 0: snr = np.var(roi_a) / np.var(roi_b)
        else: snr = np.inf
    return snr


def imshow_matplotlib(I, title:str, SNR:str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(I, cmap='gray')
    ax.set_title(title + '(SNR=%s)' %round(SNR,3))
    # ax.set_axis_off()
    fig.savefig('./midtest/img/no03_%s.png' %title)


def im_process(im_path:str, ksize:int=5, filter_list:list=['gaussian', 'mean', 'median']):
    im_dict = {}
    
    im_ori = cv.imread(im_path)    
    im_gray = cv.cvtColor(im_ori, cv.COLOR_BGR2GRAY)
    im_dict.update({"im_grayscale": im_gray})
    
    for filter in filter_list:
        im_filtered = remove_noise(im_gray, ksize, filter)
        im_dict.update({"im_%s" %filter: im_filtered})

    
    # calculate SNR of every image
    for key, val in im_dict.items():
        snr = calculate_SNR(val, im_ori, method=1)
        imshow_matplotlib(val, key, snr)
    plt.show()

    # to ilustrate the ROI region
    im_roi = cv.rectangle(im_ori.copy(), (200,10), (250,60), (0,255,0), 1)     # roi_b: background region
    im_roi = cv.rectangle(im_roi, (100,60), (130,90), (0,0,255), 1)     # roi_a: forehead region
    cv.putText(im_roi, 'ROI-B', (202,58), cv.FONT_HERSHEY_PLAIN, 0.6, (0,255,0))
    cv.putText(im_roi, 'ROI-A', (102,88), cv.FONT_HERSHEY_PLAIN, 0.6, (0,0,255))
    plt.imshow( cv.cvtColor(im_roi, cv.COLOR_BGR2RGB) )
    plt.savefig('./midtest/img/no03_im_roi.png')
    plt.show()


if __name__ == "__main__":
    im_path = './midtest/img/speckle.tif'
    im_process(im_path)
    cv.waitKey(0)