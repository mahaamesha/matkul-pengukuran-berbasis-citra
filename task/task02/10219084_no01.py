import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.spatial import distance as dist
from math import sqrt


def imshow_resized(I, winname='output', divider=1):
    h,w = I.shape[:2]
    im = cv.resize(I, (w//divider,h//divider))
    cv.imshow(winname, im)

def show_histogram(I, channels=[0], mask=None, hist_size=[256], ranges=[0,256]):
    hist = cv.calcHist(I, channels, mask, hist_size, ranges)
    plt.plot(hist)
    plt.grid(which='major', linewidth=1.2)
    plt.grid(which='minor', linewidth=0.6)
    plt.minorticks_on()
    plt.tight_layout()
    # then, call plt.show() in main program

def show_hsv_hist(I):
    h,s,v = I[:,:,0], I[:,:,1], I[:,:,2]
    plt.subplot(131), plt.title('H'), show_histogram([h], channels=[0])
    plt.subplot(132), plt.title('S'), show_histogram([s], channels=[0])
    plt.subplot(133), plt.title('V'), show_histogram([v], channels=[0])
    # then, call plt.show() in main program

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

# get centroid of every detected cnt, return it as array
def get_centroid(cnts):
    arr = []    # to store centroid(s)
    for cnt in cnts:    # iterate on every object detected
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        arr.append( (cx, cy) )
    return arr

def put_text_centroid(frame, xc, yc, text, factor=1):
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

# draw point for every centroid in centroid_array
def draw_centroid(frame, arr):
    # access pixel and change the color
    if len(arr) != 0: count_id = 1
    for (xc, yc) in arr:    # remember that, arr = [(cendtroid_id_1), (centroid_id_2), ...]
        cv.circle(frame, center=(xc, yc), radius=1, color=(0,0,255), thickness=2)
        put_text_centroid(frame, xc, yc, text="(%s, %s)" %(xc, yc), factor=1)
        count_id += 1
    return frame

def im_process(im_path='./img/filename.jpg'):
    im_ori = cv.imread(sys.path[0] + im_path)
    im_resize = cv.resize(im_ori, (960,540))
    im_hsv = cv.cvtColor(im_resize, cv.COLOR_BGR2HSV)
    im_blur = cv.GaussianBlur(im_hsv, (5,5), 5)
    
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
    print('Jumlah kontur terdeteksi: %s' %len(cnts))
    im_contoured, cnts = draw_contours(im_resize.copy(), cnts)

    # get n draw centroid of every cnt
    centroid_arr = get_centroid(cnts)
    im_centroided = draw_centroid(im_contoured, centroid_arr)
    
    return im_centroided, centroid_arr

def get_euclidian_distance(pt1=[0,0], pt2=[0,0]):
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = sqrt(dx**2 + dy**2)
    return dist

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# i want to blend 2 image
def get_final_im(im1, im2, pts1, pts2):
    im = cv.addWeighted(im1, 0.5, im2, 0.5, gamma=0.0)
    cv.line(im, pts1, pts2, (0,0,255), 1)
    (xc,yc) = midpoint(pts1, pts2)
    distance = get_euclidian_distance(pts1, pts2)
    put_text_centroid(im, int(xc), int(yc), text='d=%spx' %int(distance), factor=1)
    return im

def save_list_images(list_im=[], labels=[]):
    for i in range(len(list_im)):
        filename = sys.path[0] + './img/%s.jpg' %(labels[i])
        cv.imwrite(filename, list_im[i])


if __name__ == "__main__":
    im_pos1, centroid_arr1 = im_process(im_path='./img/pos1.jpg')
    print(centroid_arr1)
    im_pos2, centroid_arr2 = im_process(im_path='./img/pos2.jpg')
    print(centroid_arr2)

    im_final = get_final_im(im_pos1, im_pos2, centroid_arr1[0], centroid_arr2[0])
    
    save_list_images([im_pos1, im_pos2, im_final], ['im_pos1', 'im_pos2', 'im_final'])

    imshow_resized(im_pos1, 'im_pos1')
    imshow_resized(im_pos2, 'im_pos2')
    imshow_resized(im_final, 'final')
    cv.waitKey(0)