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
import pandas as pd

dir_path = os.path.dirname(__file__)
im_dir = 'citra'        # INPUT THE IMAGE DIRECTORY !!!

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
    im_blur = cv.GaussianBlur(im_gray, (7,7), 3)
    im_binary = cv.adaptiveThreshold(im_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)

    # make histogram
    # build_histogram(im_gray, filename + '_gray')
    # build_histogram(im_binary, filename + '_binary')

    # calculate the number of intensity: 0 n 255
    total_pixels = im_binary.shape[0] * im_binary.shape[1]      # h x w
    num_0_pixels = np.count_nonzero(im_binary == 0)
    percent_0_pixels = round(num_0_pixels/total_pixels*100, 3)
    num_255_pixels = np.count_nonzero(im_binary == 255)
    percent_255_pixels = round(num_255_pixels/total_pixels*100, 3)
    print( f'Percentage of 0 intensity: {percent_0_pixels}%' )
    print( f'Percentage of 255 intensity: {percent_255_pixels}%' )

    # save images
    # cv.imwrite(os.path.join(dir_path, f'results/{filename}_gray.jpg'), im_gray)
    cv.imwrite(os.path.join(dir_path, f'results/{filename}_binary.jpg'), im_binary)
    return percent_0_pixels, percent_255_pixels

def append_data_to_df(df, i, n1, n2):
    new_row = pd.DataFrame([ {
        'Dataset': i,
        'White before (%)': n1,
        'White after (%)': n2,
        'Diff': n2-n1
    } ])
    df = pd.concat([df, new_row], ignore_index=True)
    return df
    

if __name__ == '__main__':
    path_db = get_database_path()
    df = pd.DataFrame()
    for i, input_path in enumerate(path_db):
        print(f'===== DATASET {i} ==========')
        black_percentage = [None, None]     # before after
        white_percentage = [None, None]     # before after
        for file_path in input_path:
            filename = os.path.basename(file_path)
            print(f'Process the {filename} ...')
            if 'before' in filename:
                black_percentage[0], white_percentage[0] = process(file_path)
            elif 'after' in filename:
                black_percentage[1], white_percentage[1] = process(file_path)
            print()
        df = append_data_to_df(df, i, white_percentage[0], white_percentage[1])
    df = df.set_index('Dataset')
    print(df)