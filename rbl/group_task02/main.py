# Group 8 - Image Based Measurement

# Member:
# 10219084 Avima Haamesha
# 13319052 Enrico Zuriel
# 13319104 Mega Rizki Rachmannisa
# 13320003 Ananda Aikoo Mutiara K
# 13320086 Shalahudin Pasha H

import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

dir_path = os.path.dirname(__file__)
im_dir = 'citra'        # INPUT THIS !!!

def get_database_path():
    fnames = os.listdir( os.path.join(dir_path, im_dir) )
    path_db = []    # 2d matrix: n rows x 2 cols, contains before-after paths
    for i in range(0, len(fnames)-1, 2):
        path1 = os.path.join(dir_path, im_dir, fnames[i])
        path2 = os.path.join(dir_path, im_dir, fnames[i+1])
        path_db.append( [path1, path2] )
    return path_db

def build_histogram(im_single_channel, filename:str):
    hist = cv.calcHist([im_single_channel], [0], None, [256], [0, 256])
    fig, ax = plt.subplots()
    ax.plot(hist)
    ax.set_xlim([0, 256])
    fig.savefig( os.path.join(dir_path, f'{dir_path}/{filename}_histogram.jpg') )
    return fig

def process(file_path:str):
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # process the image
    im_ori = cv.imread(file_path)
    im_gray = cv.cvtColor(im_ori, cv.COLOR_BGR2GRAY)
    im_binary = cv.adaptiveThreshold(im_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)

    # make histogram
    # build_histogram(im_gray, filename + '_gray')
    # build_histogram(im_binary, filename + '_binary')

    # calculate the number of intensity: 0 n 255
    total_pixels = im_binary.shape[0] * im_binary.shape[1]      # h x w
    num_0_pixels = np.count_nonzero(im_binary == 0)
    num_255_pixels = np.count_nonzero(im_binary == 255)
    print( f'Percentage of 0 intensity: {round(num_0_pixels/total_pixels*100, 3)}%' )
    print( f'Percentage of 255 intensity: {round(num_255_pixels/total_pixels*100, 3)}%' )

    # save images
    # cv.imwrite(os.path.join(dir_path, f'img/{filename}_ori.jpg'), im_ori)
    # cv.imwrite(os.path.join(dir_path, f'img/{filename}_gray.jpg'), im_gray)
    cv.imwrite(os.path.join(dir_path, f'results/{filename}_binary.jpg'), im_binary)
    

if __name__ == '__main__':
    path_db = get_database_path()
    for input_path in path_db:
        for file_path in input_path:
            filename = os.path.basename(file_path)
            print(f'Process the {filename} ...')
            process(file_path)
            print()
