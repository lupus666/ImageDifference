import cv2 as cv
import numpy as np
import imutils
import os
from PIL import Image
from PIL import ImageChops
from matplotlib import pyplot as plt


image1 = cv.imread('./sample/0006_3.jpg')
image1 = cv.resize(image1, (960, 540))
# image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
image2 = cv.imread('./sample/0006_3_A.jpg')
image2 = cv.resize(image2, (960, 540))

# image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
[height, width, channels] = image1.shape
print(height, width)
# image2 = cv.resize(image2, (457, 395))
image3 = cv.subtract(image2, image1)
# image3 = image3 - np.mean(image3)
image3[image3 < 0] = 0
# image3 = cv.cvtColor(image3, cv.COLOR_BGR2GRAY)
cv.imshow("1", image3)
image3 = cv.cvtColor(image3, cv.COLOR_BGR2GRAY)
cv.imshow("1.1", image3)


def colorDiff(img1, img2):
    b1, g1, r1 = cv.split(img1)
    b2, g2, r2 = cv.split(img2)

    diff1 = cv.absdiff(b1, b2)
    diff2 = cv.absdiff(g1, g2)
    diff3 = cv.absdiff(r1, r2)
    # 为凸显红蓝色差，计算第四个色差
    diff4 = cv.absdiff(b1 - r1, b2 - r2)
    # 各个色差加权求和
    res = diff1 * 0.35 + diff2 * 0.2 + diff3 * 0.35 + diff4 * 0.1
    mean = np.mean(res)
    res_n = res - mean
    res_n[res_n < 0] = 0

    return res_n


diff = colorDiff(image1, image2).astype(np.uint8)
cv.imshow("2", diff)
# diff = image3


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
# thresh = diff
thresh = deEdge(diff, xthresh=0.2, ythresh=0.2)
cv.imshow("3", thresh)

# Gauss filtering
size = int(height*0.005) + (int(height*0.005) - 1) % 2
print("SIZE :", size)
blur = cv.GaussianBlur(thresh, (size, size), 0)
cv.imshow("3.5", blur)

# Morphology operation
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
thresh = cv.morphologyEx(blur, cv.MORPH_CLOSE, kernel, iterations=15)
cv.imshow("3.9", thresh)

# Gauss filtering
size = int(height*0.014) + (int(height*0.014) - 1) % 2
print("SIZE :", size)
blur = cv.GaussianBlur(thresh, (3, 3), 0)
blur = thresh
cv.imshow("4", blur)


# Threshold filtering
value, thresh = cv.threshold(blur, 10, 255,  cv.THRESH_OTSU | cv.THRESH_TOZERO)
print(value)
cv.imshow("5", thresh)

# Get contours
cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] if imutils.is_cv3() else cnts[0]
print(len(cnts))


# def area(c):
#     import cv2
#     x, y, w, h = cv2.boundingRect(c)
#     return w*h
#
#
# # Sort by area
# cnts.sort(key=lambda x: area(x))
# cnts.reverse()
# Draw rectangle the first iteration
blank = np.zeros([int(height), int(width), 3], np.uint8)
edgeThresh = 0.04
i = 0
for c in cnts:
    x, y, w, h = cv.boundingRect(c)
    # print(x, y, w, h)
    if w > edgeThresh * width and h > edgeThresh * height:
        i += 1
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        cv.rectangle(blank, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv.imshow("6", blank)
blank = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
cnts = cv.findContours(blank, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] if imutils.is_cv3() else cnts[0]
print(len(cnts))
for c in cnts:
    x, y, w, h = cv.boundingRect(c)
    if w > 0.08 * width and h > 0.08 * height:
        print("True ", x, y, x + w, y + h)

print("I:{}".format(i))

# numpy
# image1f = np.fft.fft2(image1)
# image1fs = np.fft.fftshift(image1f)
# image1fsms = 20*np.log(np.abs(image1fs))
# image1fsms = image1fsms.astype(np.int8)
# cv.imshow('mat', image1fsms)
# print(image1.shape)
# print(image2.shape)
#
# # openCV
# dft1 = cv.dft(np.float32(image1), flags=cv.DFT_COMPLEX_OUTPUT)
# dft1_shift = np.fft.fftshift(dft1)
# magnitude_spectrum = 20*np.log(cv.magnitude(dft1_shift[:, :, 0], dft1_shift[:, :, 1]))
# plt.subplot(121), plt.imshow(image1, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()
# magnitude_spectrum = magnitude_spectrum.astype(np.int8)
# cv.imshow('magnitude_spectrum', magnitude_spectrum)


# [height, width, channels] = image2.shape
# image3 = np.zeros((height, width, channels))
#
# image3 = cv.subtract(image1, image2)
# cv.imshow('mat', image3)
cv.waitKeyEx(0)
