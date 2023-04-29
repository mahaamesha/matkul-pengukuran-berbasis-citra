import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

dir_path = os.path.dirname(__file__)

def build_histogram(im_single_channel, filename:str):
    hist = cv.calcHist([im_single_channel], [0], None, [256], [0, 256])
    fig, ax = plt.subplots()
    ax.plot(hist)
    ax.set_xlim([0, 256])
    fig.savefig( os.path.join(dir_path, f'img/{filename}_histogram.jpg') )
    return fig

def process(file_path:str):
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # process the image
    im_ori = cv.imread(file_path)
    im_gray = cv.cvtColor(im_ori, cv.COLOR_BGR2GRAY)
    im_binary = cv.adaptiveThreshold(im_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)

    # make histogram
    build_histogram(im_gray, filename + '_gray')
    build_histogram(im_binary, filename + '_binary')

    # calculate the number of intensity: 0 n 255
    num_0_pixels = np.count_nonzero(im_binary == 0)
    num_255_pixels = np.count_nonzero(im_binary == 255)
    print(f'Number of 0 intensity: {num_0_pixels}')
    print(f'Number of 255 intensity: {num_255_pixels}')

    # save images
    cv.imwrite(os.path.join(dir_path, f'img/{filename}_ori.jpg'), im_ori)
    cv.imwrite(os.path.join(dir_path, f'img/{filename}_gray.jpg'), im_gray)
    cv.imwrite(os.path.join(dir_path, f'img/{filename}_binary.jpg'), im_binary)
    

if __name__ == '__main__':
    file_paths = [
        os.path.join( dir_path, 'img/kering.jpg' ),
        os.path.join( dir_path, 'img/normal.jpg' ),
    ]
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        print(f'Process the {filename} ...')
        process(file_path)
        print()
