import cv2 as cv
import numpy as np
import imutils
import os
from PIL import Image
from PIL import ImageChops
from matplotlib import pyplot as plt

# 多次测试
image1 = cv.imread('./sample/in000113.jpg')
image1 = cv.resize(image1, (320, 320))
# image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
image2 = cv.imread('./sample/in000113_A.jpg')
image2 = cv.resize(image2, (320, 320))

# image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
[height, width, channels] = image1.shape
print(height, width)
# image2 = cv.resize(image2, (457, 395))
image3 = cv.subtract(image2, image1)
# image3 = image3 - np.mean(image3)
image3[image3 < 0] = 0
# image3 = cv.cvtColor(image3, cv.COLOR_BGR2GRAY)
# cv.imshow("1", image3)
image3 = cv.cvtColor(image3, cv.COLOR_BGR2GRAY)
# cv.imshow("1.1", image3)


def colorDiff(img1, img2):
    b1, g1, r1 = cv.split(img1)
    b2, g2, r2 = cv.split(img2)

    diff1 = cv.absdiff(b1, b2)
    diff2 = cv.absdiff(g1, g2)
    diff3 = cv.absdiff(r1, r2)
    # 为凸显红蓝色差，计算第四个色差
    diff4 = cv.absdiff(b1 - r1, b2 - r2)
    # 各个色差加权求和
    res = diff1 * 0.33 + diff2 * 0.33 + diff3 * 0.33
    mean = np.mean(res)
    res_n = res - mean
    res_n[res_n < 0] = 0

    return res_n


# diff = colorDiff(image1, image2).astype(np.uint8)
# cv.imshow("2", diff)
# # diff = image3


def deEdge(img, xthresh=1.0, ythresh=1.0):
    """
    Remove/Blur edge
    :param img:
    :param xthresh: Can be adjusted
    :param ythresh: Can be adjusted
    :return:
    """
    import numpy as np
    import cv2 as cv

    edge1 = cv.Sobel(img, cv.CV_16S, 0, 1)
    edge1 = cv.convertScaleAbs(edge1)

    edge2 = cv.Sobel(img, cv.CV_16S, 1, 0)
    edge2 = cv.convertScaleAbs(edge2)

    deedgedImg = img.astype(np.float32) - edge1.astype(np.float)*xthresh - edge2.astype(np.float)*ythresh
    deedgedImg[deedgedImg < 0] = 0
    deedgedImg = deedgedImg.astype(np.uint8)

    return deedgedImg


# Remove little block
image1_thresh = deEdge(image1, xthresh=0.2, ythresh=0.2)
cv.imshow("Image 1 deEdge", image1_thresh)
image1_thresh = image1


kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
image1_thresh = cv.morphologyEx(image1_thresh, cv.MORPH_OPEN, kernel, iterations=1)

cv.imshow("Image 1 morphology", image1_thresh)

size = int(height*0.014) + (int(height*0.014) - 1) % 2
print("SIZE :", size)
image1_thresh = cv.GaussianBlur(image1_thresh, (size, size), 0)
cv.imshow("Image 1 gaussianBlur", image1_thresh)

image2_thresh = deEdge(image2, xthresh=0.2, ythresh=0.2)
cv.imshow("Image 2 deEdge", image2_thresh)
image2_thresh = image2

image2_thresh = cv.morphologyEx(image2_thresh, cv.MORPH_OPEN, kernel, iterations=1)

cv.imshow("Image 2 morphology", image2_thresh)

image2_thresh = cv.GaussianBlur(image2_thresh, (size, size), 0)
cv.imshow("Image 2 gaussianBlur", image2_thresh)


diff = colorDiff(image1_thresh, image2_thresh)
cv.imshow("Image difference", diff)
# diff2 = cv.subtract(image1_thresh, image2_thresh)
# cv.imshow("DIFFERENCE", diff2)
# diff.astype(np.uint8)
diff = diff.astype(np.uint8)
cv.imshow("Uint8", diff)
# diff = cv.GaussianBlur(diff, (size, size), 0)
# cv.imshow("Diff gaussianBlur", diff)
# value, thresh = cv.threshold(diff, 30, 255, cv.THRESH_TRIANGLE | cv.THRESH_TOZERO_INV)
thresh = cv.adaptiveThreshold(diff, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 3, 2.5)  # C can be adaptive
cv.imshow("Threshold", thresh)
# print(value)


cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] if imutils.is_cv3() else cnts[0]
print(len(cnts))

blank = np.zeros([int(height), int(width), 3], np.uint8)
edgeThresh = 0.03
# i = 0
# adaptive by cnts
for c in cnts:
    x, y, w, h = cv.boundingRect(c)
    # print(x, y, w, h)
    # if w > 0.005 * width and h > 0.005 * height:
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    cv.rectangle(blank, (x1, y1), (x2, y2), (0, 0, 255), thickness=0)  # -1 - 3

cv.imshow("Blank", blank)


blank2 = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
blank = np.zeros([int(height), int(width), 3], np.uint8)
cnts = cv.findContours(blank2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] if imutils.is_cv3() else cnts[0]
# i = 0
for c in cnts:
    x, y, w, h = cv.boundingRect(c)
    # print(x, y, w, h)
    if w > edgeThresh * width and h > edgeThresh * height:
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        cv.rectangle(blank, (x1, y1), (x2, y2), (0, 0, 255), 0)

cv.imshow("Blank2", blank)
print(blank.shape)

blank2 = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
cnts = cv.findContours(blank2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] if imutils.is_cv3() else cnts[0]
print("len of cnts: ", len(cnts))
for c in cnts:
    x, y, w, h = cv.boundingRect(c)
    if w > 0.06 * width and h > 0.06 * height:
        print("True ", x, y, x + w, y + h)
        cv.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv.imshow("Final", image1)
# print("I:{}".format(i))

cv.waitKeyEx(0)
