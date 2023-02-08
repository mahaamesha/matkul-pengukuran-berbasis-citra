import cv2 as cv
import numpy as np
import sys
from scipy.spatial import distance as dist
from skimage.util import random_noise

def imshow_resized(I, winname='output', divider=1):
    h,w = I.shape[:2]
    im = cv.resize(I, (w//divider,h//divider))
    cv.imshow(winname, im)

def add_salt_pepper(I, var=0.05**2):
    noise_img = random_noise(I, mode='gaussian', var=var)
    noise_img = (255*noise_img).astype(np.uint8)          
    return I

# salt n paper noise removal
def remove_noise(I, ksize, filter='median/gaussian'):
    if filter == 'median': im_filtered = cv.medianBlur(I, ksize)
    elif filter == 'gaussian': im_filtered = cv.GaussianBlur(I, (ksize, ksize), 2)
    return im_filtered

# i want to rearrange: topleft, topright, btmright, btmleft
def reorder_box(pts):
	xSorted = pts[np.argsort(pts[:, 0]), :]
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	return np.array([tl, tr, br, bl])

def get_boxes(cnts):
    boxes = []      # array to store box for every contour
    for cnt in cnts:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        box = reorder_box(box)      # reorder 4 points in box
        boxes.append(box)
    return boxes

def draw_contours(I, cnts):
    boxes = get_boxes(cnts)
    cv.drawContours(I, boxes, -1, (0,255,255), 1)
    return I, boxes

def put_text(frame, xc, yc, text, factor=1):
    dist = 20*factor
    org = (xc-dist-dist, yc-dist)
    fontFace = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5*factor
    color = (0,255,0)
    thickness = 1*factor
    text_color_bg = (0,0,0)
    # background
    text_size, _ = cv.getTextSize(text, fontFace, fontScale, thickness)
    text_w, text_h = text_size
    cv.rectangle(frame, org, (org[0]+text_w, org[1]-text_h), text_color_bg, -1)
    cv.putText(frame, text, org, fontFace, fontScale, color, thickness, cv.LINE_AA)

def im_process(im_path='./img/filename.jpg', ksize=5, filter='median/gaussian'):
    im_ori = cv.imread(sys.path[0] + im_path)
    im_resize = cv.resize(im_ori, (960,540))
    im_noise = add_salt_pepper(im_resize, var=2)
    im_adjust = cv.convertScaleAbs(im_noise, alpha=1.5, beta=60)  # alpha: contrast [0,1], beta: brightness [-127,127]
    im_hsv = cv.cvtColor(im_adjust, cv.COLOR_BGR2HSV)
    im_blur = remove_noise(im_hsv, ksize, filter)
    
    # from the hsv histogram, i know the lower n upper hsv
    # show_hsv_hist(im_hsv), plt.show()   # hide this later
    lower_hsv, upper_hsv = np.array([50,64,64]), np.array([70,255,255])
    mask = cv.inRange(im_blur, lower_hsv, upper_hsv)
    
    # remove little area
    kernel = np.ones((5,5), np.uint8)
    im_eroded = cv.erode(mask, kernel, 10)
    im_dilated = cv.dilate(im_eroded, kernel, 10)
    
    im_edged = cv.Canny(im_dilated, 0, 255)
    cnts, hierarchy = cv.findContours(im_edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # print('Jumlah kontur terdeteksi: %s' %len(cnts))
    im_contoured, cnts = draw_contours(im_adjust.copy(), cnts)

    # area n perimeter
    area = cv.contourArea(cnts[0])
    perimeter = cv.arcLength(cnts[0], 1)
    text = 'area=%spx^2, perimeter=%spx' %(int(area), int(perimeter))
    print(text)
    put_text(im_contoured, xc=50, yc=50, text=text)
    
    # hconcat im_noise n im_processed
    im_final = cv.hconcat([im_noise, im_contoured])

    return im_final

def save_list_images(list_im=[], labels=[]):
    for i in range(len(list_im)):
        filename = sys.path[0] + './img/%s.jpg' %(labels[i])
        cv.imwrite(filename, list_im[i])


if __name__ == "__main__":
    im_path = './img/DSC_2705.jpg'
    arr_ksize = [5, 51, 111]
    arr_filter = ['median', 'gaussian']
    
    arr_im = []
    labels = []
    for filter in arr_filter:
        for ksize in arr_ksize:
            print('%s_%sx%s' %(filter, ksize, ksize))
            im = im_process(im_path, ksize, filter)
            labels.append('%s_%sx%s' %(filter, ksize, ksize))
            arr_im.append(im)

            # imshow_resized(im, '%s_%sx%s' %(filter, ksize, ksize))
            # cv.waitKey(0)

    save_list_images(arr_im, labels)
    