import cv2 as cv
import imutils
import numpy as np
import os
import time
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim


class ImageDataSet(object):
    def __init__(self, root):
        self.root = root
        self.image = list(sorted(os.listdir(root)))

    def __len__(self):
        return len(self.image) // 2

    def __getitem__(self, item):
        basePath = os.path.join(self.root, self.image[2*item])
        targetPath = os.path.join(self.root, self.image[2*item + 1])
        baseImage = cv.imread(basePath)
        targetImage = cv.imread(targetPath)
        basename = self.image[2*item].split(".")[0]

        return basename, baseImage, targetImage


class DetectionModel(object):
    def __init__(self, edgeX=1.0, edgeY=1.0, threshold=0, morphologyIter=15, rectEdgeThresh=0.03, calibrate=False, nFeatures=2000):
        self.threshold = threshold
        self.edgeX = edgeX
        self.edgeY = edgeY
        self.morphologyIter = morphologyIter
        self.rectEdgeThresh = rectEdgeThresh
        self.calibrate = calibrate

    def detection(self, dataset, respath):
        total_time = 0
        callable_time = 0
        diff_time = 0
        deEdge_time = 0
        morph_time = 0
        filter_time = 0
        drawRect_time = 0
        for i in range(0, len(dataset)):
            name, baseImage, targetImage = dataset[i]

            # Calibration
            callable_start_time = time.time()
            if self.calibrate:
                pass
            callable_end_time = time.time()
            callable_time += callable_end_time - callable_start_time

            # Color Difference
            diff_start_time = time.time()
            diff = self.colorDifference(baseImage, targetImage).astype(np.uint8)
            diff_end_time = time.time()
            diff_time += diff_end_time - diff_start_time

            # DeEdge
            deEdge_start_time = time.time()
            thresh = self.deEdge(diff)
            deEdge_end_time = time.time()
            deEdge_time += deEdge_end_time - deEdge_start_time

            # Morphology operation
            morphology_start_time = time.time()
            thresh = self.morphology(thresh)
            morphology_end_time = time.time()
            morph_time += morphology_end_time - morphology_start_time

            # Gauss & Thresh filtering
            filter_start_time = time.time()
            value, thresh = self.filter(thresh)
            filter_end_time = time.time()
            filter_time += filter_end_time - filter_start_time

            # Draw rectangle
            drawRectangle_start_time = time.time()
            items = self.drawRectangle(thresh)
            drawRectangle_end_time = time.time()
            drawRect_time += drawRectangle_end_time - drawRectangle_start_time

            # Output
            f = open(os.path.join(respath, name + '.txt'), 'w')
            for item in items:
                strs = " ".join(item)
                strs = "True 1.0 " + strs + '\n'
                f.write(strs)

            f.close()
            print("{}/{}".format(i, len(dataset) - 1))

        total_time = callable_time + diff_time + deEdge_time + morph_time + filter_time +drawRect_time
        print("Total time: {}".format(total_time))
        print("Calibration time: {}".format(callable_time))
        print("Diff time: {}".format(diff_time))
        print("DeEdge time: {}".format(deEdge_time))
        print("Morph time: {}".format(morph_time))
        print("Filter time: {}".format(filter_time))
        print("DrawRect time: {}".format(drawRect_time))

    def calibrate(self, baseImage, targetImage):
        pass

    def colorDifference(self, image1, image2):
        b1, g1, r1 = cv.split(image1)
        b2, g2, r2 = cv.split(image2)

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

    def deEdge(self, image):
        edge1 = cv.Sobel(image, cv.CV_16S, 0, 1)
        edge1 = cv.convertScaleAbs(edge1)

        edge2 = cv.Sobel(image, cv.CV_16S, 1, 0)
        edge2 = cv.convertScaleAbs(edge2)

        deedgedImg = image.astype(np.float32) - edge1.astype(np.float) * self.edgeX - edge2.astype(np.float) * self.edgeY
        deedgedImg[deedgedImg < 0] = 0
        deedgedImg = deedgedImg.astype(np.uint8)

        return deedgedImg

    def morphology(self, image):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        thresh = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=self.morphologyIter)

        return thresh

    def filter(self, image):
        # Gauss filtering
        [height, width] = image.shape
        size = int(height * 0.014) + (int(height * 0.014) - 1) % 2
        blur = cv.GaussianBlur(image, (3, 3), 0)
        # blur = image
        # Threshold filtering
        value, thresh = cv.threshold(blur, 20, 255, cv.THRESH_OTSU | cv.THRESH_TOZERO)

        return value, thresh

    def drawRectangle(self, image):

        [height, width] = image.shape

        cnts1 = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts1 = cnts1[1] if imutils.is_cv3() else cnts1[0]

        res = []

        blank = np.zeros([int(height), int(width), 3], np.uint8)
        for c in cnts1:
            x, y, w, h = cv.boundingRect(c)
            # print(x, y, w, h)
            if w > self.rectEdgeThresh * width and h > self.rectEdgeThresh * height:
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                cv.rectangle(blank, (x1, y1), (x2, y2), (0, 0, 255), cv.FILLED)

        blank2 = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
        cnts2 = cv.findContours(blank2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts2 = cnts2[1] if imutils.is_cv3() else cnts2[0]
        for c in cnts2:
            x, y, w, h = cv.boundingRect(c)
            if w > 0.08 * width and h > 0.08 * height:
                print(x, y, x + w, y + h)
                item = [str(x), str(y), str(x + w), str(y + h)]
                res.append(item)

        return res


dataset = ImageDataSet("./sample/")
model = DetectionModel(0.2, 0.2)
model.detection(dataset, './sample_res/')

