import cv2 as cv
import imutils
import numpy as np
import os
import time
import math


class ImageDataSet(object):
    def __init__(self, root):
        self.root = root
        self.image = list(sorted(os.listdir(root)))

    def __len__(self):
        return len(self.image) // 2

    def __getitem__(self, item):
        basePath = os.path.join(self.root, self.image[2 * item])
        targetPath = os.path.join(self.root, self.image[2 * item + 1])
        baseImage = cv.imread(basePath)
        targetImage = cv.imread(targetPath)
        basename = self.image[2 * item].split(".")[0]

        return basename, baseImage, targetImage


class Calibrator(object):
    def __init__(self, nFeature):
        self.nFeature = nFeature

    def sift_kp(self, image, number, method):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 颜色空间转换
        if method == 'sift':
            sift = cv.xfeatures2d.SIFT_create(self.nFeature)
        else:
            sift = cv.xfeatures2d.SURF_create()
        kps, des = sift.detectAndCompute(image, None)
        kp_image = cv.drawKeypoints(gray_image, kps, None)  # 绘制关键点的函数
        # cv.imshow('kp'+number, kp_image)
        return kp_image, kps, des

    def get_good_match(self, des1, des2):
        matcher = cv.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for (m, n) in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)
        return good

    def siftImageAlignment(self, img1, img2, method='sift'):
        _, kp1, des1 = self.sift_kp(img1, '1', method)
        _, kp2, des2 = self.sift_kp(img2, '2', method)
        goodMatch = self.get_good_match(des1, des2)
        img3 = cv.drawMatches(img1, kp1, img2, kp2, goodMatch, flags=2, outImg=None)
        # cv.imshow('match', img3)
        imgOut = img2
        H = 0
        status = 0
        if len(goodMatch) > 4:
            ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ransacReprojThreshold = 1.86
            H, status = cv.findHomography(ptsA, ptsB, cv.RANSAC, ransacReprojThreshold)
            imgOut = cv.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]),
                                        flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_REPLICATE)
            # if method == 'sift':
            #     if 0.08 >= H[0, 2] >= -0.08 and 0.08 >= H[1, 2] >= -0.08:
            #         imgOutTemp, tempH, tempStatus = self.siftImageAlignment(img1, imgOut, 'other')
            #         if not (1 >= tempH[0, 2] >= -1 and 1 >= tempH[1, 2] >= -1):
            #             imgOut = imgOutTemp
            #             H = tempH
            #             status = tempStatus

        return imgOut, H, status


class DetectionModel(object):                                   # 2
    def __init__(self, edgeX=1.0, edgeY=1.0, threshold=0, morphologyIter=1, rectEdgeThresh=0.03, calibrate=True,
                 nFeatures=2000, showImage=False, pause=False, normalBlurSize=0.015, firstBlurSize=0.03, secondBlurSize=0.03, normalThreshSize=3,
                 normalThreshC=2.5, hugeThreshSize=3, hugeThreshC=2.5, hugeImageThresh=360000):
        self.threshold = threshold
        self.edgeX = edgeX
        self.edgeY = edgeY
        self.morphologyIter = morphologyIter
        self.rectEdgeThresh = rectEdgeThresh
        self.calibrate = calibrate
        self.nFeatures = nFeatures
        self.showImage = showImage
        self.pause = pause
        self.firstBlurSize = firstBlurSize
        self.secondBlurSize = secondBlurSize
        self.normalBlurSize = normalBlurSize
        self.normalThreshSize = normalThreshSize
        self.normalThreshC = normalThreshC
        self.hugeThreshSize = hugeThreshSize
        self.hugeThreshC = hugeThreshC
        self.hugeImageThresh = hugeImageThresh
        self.total_time = 0
        self.resize_time = 0
        self.callable_time = 0
        self.diff_time = 0
        self.deEdge_time = 0
        self.morph_time = 0
        self.filter_time = 0
        self.drawRect_time = 0
        self.gaussian_time = 0
        self.respath = ""
        self.flag = False
        self.originBaseImage = None
        self.originTargetImage = None

    def detection(self, dataset, respath):
        self.respath = respath
        for i in range(0, len(dataset)):
            print("{}/{}:".format(i + 1, len(dataset)))

            ishuge = False
            name, baseImage, targetImage = dataset[i]
            # if self.showImage:
            #     cv.imshow('origin1', baseImage)
            #     cv.imshow('origin2', targetImage)
            height, width, channels = baseImage.shape

            self.originBaseImage = baseImage
            self.originTargetImage = targetImage

            resize_height = 320
            resize_width = 320
            if height * width > self.hugeImageThresh:
                ishuge = True
                resize_height = 320
                resize_width = 320
                self.morphologyIter = 1

            # Calibration
            callable_start_time = time.time()
            if self.calibrate:
                targetImage = self.doCalibrate(baseImage, targetImage)
            callable_end_time = time.time()
            self.callable_time += callable_end_time - callable_start_time
            # if self.showImage:
            #     cv.imshow('calibrate', targetImage)

            # DeEdge
            if ishuge:
                deEdge_start_time = time.time()
                baseImage = self.deEdge(baseImage)
                targetImage = self.deEdge(targetImage)
                deEdge_end_time = time.time()
                self.deEdge_time += deEdge_end_time - deEdge_start_time

            # Gaussian filtering
            if ishuge:
                gaussian_start_time = time.time()
                baseImage = self.gaussionFilter(baseImage, ishuge, self.firstBlurSize)    # 0.25
                targetImage = self.gaussionFilter(targetImage, ishuge, self.firstBlurSize)
                gaussian_end_time = time.time()
                self.gaussian_time += gaussian_end_time - gaussian_start_time

            # Resize
            resize_start_time = time.time()
            baseImage, height_ratio1, width_ratio1 = self.resize(image=baseImage, dsize=(resize_width, resize_height))
            targetImage, height_ratio2, width_ratio2 = self.resize(image=targetImage,
                                                                   dsize=(resize_width, resize_height))
            resize_end_time = time.time()
            self.resize_time += resize_end_time - resize_start_time
            if self.showImage:
                cv.imshow('resize1', baseImage)
                cv.imshow('resize2', targetImage)

            # DeEdge
            # if ishuge:
            #     deEdge_start_time = time.time()
            #     baseImage = self.shape(baseImage)
            #     targetImage = self.shape(targetImage)
            #     deEdge_end_time = time.time()
            #     deEdge_time += deEdge_end_time - deEdge_start_time
            # baseImage = resizebaseImage
            # targetImage = resizeTargetImage

            # Morphology operation
            morphology_start_time = time.time()
            baseImage = self.morphology(baseImage, 3, ishuge)
            targetImage = self.morphology(targetImage, 3, ishuge)
            morphology_end_time = time.time()
            self.morph_time += morphology_end_time - morphology_start_time
            # if self.showImage:
            #     cv.imshow('morphology1', baseImage)
            #     cv.imshow('morphology2', targetImage)

            # Gaussian filtering
            gaussian_start_time = time.time()
            baseImage = self.gaussionFilter(baseImage, ishuge, 0.03)
            targetImage = self.gaussionFilter(targetImage, ishuge, 0.03)
            gaussian_end_time = time.time()
            self.gaussian_time += gaussian_end_time - gaussian_start_time
            if self.showImage:
                cv.imshow('filter1', baseImage)
                cv.imshow('filter2', targetImage)

            # Color Difference
            diff_start_time = time.time()
            diff = self.colorDifference(baseImage, targetImage).astype(np.uint8)
            diff_end_time = time.time()
            self.diff_time += diff_end_time - diff_start_time
            if self.showImage:
                cv.imshow('diff', diff)

            mean = np.mean(diff)
            print("     ", math.ceil(mean))
            print("     Diff mean:", mean)
            diff[diff <= mean] += 2
            # var = np.var(diff)
            # mean = np.mean(diff)
            # std = np.std(diff)
            # print("     Variance:   ", var)
            # print("     Mean:       ", mean)
            # print("     Std:        ", std)
            # print("     var*mean:   ", var*mean)
            # print("     var/mean:   ", var/mean)
            # print("     std*mean:   ", std*mean)
            # print("     std/mean:   ", std/mean)

            # Gaussian filtering
            # if ishuge:
            #     gaussian_start_time = time.time()
            #     diff = self.gaussionFilter(diff, ishuge, self.secondBlurSize)
            #     # targetImage = self.gaussionFilter(targetImage, ishuge, self.secondBlurSize)
            #     gaussian_end_time = time.time()
            #     self.gaussian_time += gaussian_end_time - gaussian_start_time
            # # if self.showImage:
            #     # cv.imshow('filter1', baseImage)
            #     # cv.imshow('filter2', targetImage)

            # Thresh filtering
            filter_start_time = time.time()
            thresh = self.thresholdFilter(diff, ishuge)
            filter_end_time = time.time()
            self.filter_time += filter_end_time - filter_start_time
            if self.showImage:
                cv.imshow('thresh', thresh)

            # if ishuge:
            #     morphology_start_time = time.time()
            #     thresh = self.morphology(thresh, 5, ishuge)
            #     morphology_end_time = time.time()
            #     self.morph_time += morphology_end_time - morphology_start_time
            #     if self.showImage:
            #         cv.imshow('mor2', thresh)

            # Draw rectangle
            drawRectangle_start_time = time.time()
            items = self.drawRectangle(thresh, (height_ratio1, width_ratio1), baseImage, name, ishuge)
            drawRectangle_end_time = time.time()
            self.drawRect_time += drawRectangle_end_time - drawRectangle_start_time

            # Output
            f = open(os.path.join(respath, name + '.txt'), 'w')
            for item in items:
                strs = " ".join(item)
                strs = "True 1.0 " + strs + '\n'
                f.write(strs)

            f.close()

            if self.pause:
                cv.waitKey(2000)

        self.total_time = self.callable_time + self.diff_time + self.deEdge_time + self.morph_time + self.filter_time + self.drawRect_time
        print("Total time: {}".format(self.total_time))
        print("Calibration time: {}".format(self.callable_time))
        print("Diff time: {}".format(self.diff_time))
        print("DeEdge time: {}".format(self.deEdge_time))
        print("Morph time: {}".format(self.morph_time))
        print("Filter time: {}".format(self.filter_time))
        print("DrawRect time: {}".format(self.drawRect_time))

    def onceDetection(self, baseImage, targetImage, name):
        ishuge = False
        # if self.showImage:
        #     cv.imshow('origin1', baseImage)
        #     cv.imshow('origin2', targetImage)
        height, width, channels = baseImage.shape

        resize_height = 320
        resize_width = 320
        if height * width > self.hugeImageThresh:
            ishuge = True
            resize_height = 320
            resize_width = 320
            self.morphologyIter = 1

        # Calibration
        callable_start_time = time.time()
        if self.calibrate:
            targetImage = self.doCalibrate(baseImage, targetImage)
        callable_end_time = time.time()
        self.callable_time += callable_end_time - callable_start_time
        # if self.showImage:
        #     cv.imshow('calibrate', targetImage)

        # DeEdge
        if ishuge:
            deEdge_start_time = time.time()
            baseImage = self.shape(baseImage)
            targetImage = self.shape(targetImage)
            deEdge_end_time = time.time()
            self.deEdge_time += deEdge_end_time - deEdge_start_time

        # Gaussian filtering
        if ishuge:
            gaussian_start_time = time.time()
            baseImage = self.gaussionFilter(baseImage, ishuge, self.firstBlurSize)  # 0.25
            targetImage = self.gaussionFilter(targetImage, ishuge, self.firstBlurSize)
            gaussian_end_time = time.time()
            self.gaussian_time += gaussian_end_time - gaussian_start_time

        # Resize
        resize_start_time = time.time()
        baseImage, height_ratio1, width_ratio1 = self.resize(image=baseImage, dsize=(resize_width, resize_height))
        targetImage, height_ratio2, width_ratio2 = self.resize(image=targetImage,
                                                               dsize=(resize_width, resize_height))
        resize_end_time = time.time()
        self.resize_time += resize_end_time - resize_start_time
        if self.showImage:
            cv.imshow('resize1', baseImage)
            cv.imshow('resize2', targetImage)

        # DeEdge
        # if ishuge:
        #     deEdge_start_time = time.time()
        #     baseImage = self.shape(baseImage)
        #     targetImage = self.shape(targetImage)
        #     deEdge_end_time = time.time()
        #     deEdge_time += deEdge_end_time - deEdge_start_time
        # baseImage = resizebaseImage
        # targetImage = resizeTargetImage

        # Morphology operation
        morphology_start_time = time.time()
        baseImage = self.morphology(baseImage)
        targetImage = self.morphology(targetImage)
        morphology_end_time = time.time()
        self.morph_time += morphology_end_time - morphology_start_time
        # if self.showImage:
        #     cv.imshow('morphology1', baseImage)
        #     cv.imshow('morphology2', targetImage)

        # Gaussian filtering
        gaussian_start_time = time.time()
        baseImage = self.gaussionFilter(baseImage, ishuge, self.secondBlurSize)
        targetImage = self.gaussionFilter(targetImage, ishuge, self.secondBlurSize)
        gaussian_end_time = time.time()
        self.gaussian_time += gaussian_end_time - gaussian_start_time
        # if self.showImage:
        #     cv.imshow('filter1', baseImage)
        #     cv.imshow('filter2', targetImage)

        # Color Difference
        diff_start_time = time.time()
        diff = self.colorDifference(baseImage, targetImage).astype(np.uint8)
        diff_end_time = time.time()
        self.diff_time += diff_end_time - diff_start_time
        if self.showImage:
            cv.imshow('diff', diff)

        # Highlight  TODO NEW
        mean = np.mean(diff)
        diff[diff <= mean] += 2
        # gaussian_start_time = time.time()
        # diff = self.medianFilter(diff, ishuge)
        # # targetImage = self.gaussionFilter(targetImage, ishuge)
        # gaussian_end_time = time.time()
        # gaussian_time += gaussian_end_time - gaussian_start_time

        # Thresh filtering
        filter_start_time = time.time()
        thresh = self.thresholdFilter(diff, ishuge)
        filter_end_time = time.time()
        self.filter_time += filter_end_time - filter_start_time
        if self.showImage:
            cv.imshow('thresh', thresh)



        # Draw rectangle
        drawRectangle_start_time = time.time()
        items = self.drawRectangle(thresh, (height_ratio1, width_ratio1), baseImage, name, ishuge)
        drawRectangle_end_time = time.time()
        self.drawRect_time += drawRectangle_end_time - drawRectangle_start_time

        # Output
        return items

    def doCalibrate(self, baseImage, targetImage):
        calibrator = Calibrator(self.nFeatures)
        result, _, _ = calibrator.siftImageAlignment(baseImage, targetImage)
        return result

    def resize(self, dsize, image):
        [height, width, channels] = image.shape
        height_ratio = height / dsize[1]
        width_ratio = width / dsize[0]
        return cv.resize(image, dsize), height_ratio, width_ratio

    def colorDifference(self, image1, image2):
        b1, g1, r1 = cv.split(image1)
        b2, g2, r2 = cv.split(image2)

        diff1 = cv.absdiff(b1, b2)
        diff2 = cv.absdiff(g1, g2)
        diff3 = cv.absdiff(r1, r2)
        # 为凸显红蓝色差，计算第四个色差
        diff4 = cv.absdiff(r1 - b1, r2 - b2)
        # 各个色差加权求和
        res = diff1 * 0.33 + diff2 * 0.33 + diff3 * 0.33 + diff4 * 0.01
        mean = np.mean(res)
        res_n = res - mean
        res_n[res_n < 0] = 0

        return res_n

    def deEdge(self, image):
        edge1 = cv.Sobel(image, cv.CV_16S, 0, 1)
        edge1 = cv.convertScaleAbs(edge1)

        edge2 = cv.Sobel(image, cv.CV_16S, 1, 0)
        edge2 = cv.convertScaleAbs(edge2)

        deedgedImg = image.astype(np.float32) - edge1.astype(np.float) * self.edgeX - edge2.astype(
            np.float) * self.edgeY
        deedgedImg[deedgedImg < 0] = 0
        deedgedImg = deedgedImg.astype(np.uint8)

        return deedgedImg

    def shape(self, image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        shapeImage = cv.filter2D(image, -1, kernel=kernel)
        return shapeImage

    def morphology(self, image, size=3, ishuge=False):
        type = cv.MORPH_OPEN
        if ishuge:
            type = cv.MORPH_CLOSE
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))                        # cv.BORDER_REFLECT
        thresh = cv.morphologyEx(image, type, kernel, iterations=self.morphologyIter, borderType=cv.BORDER_CONSTANT)

        return thresh

    def thresholdFilter(self, image, ishuge=False):
        # Threshold filtering C could be adaptive
        ksize = self.normalThreshSize
        C = self.normalThreshC
        if ishuge:
            ksize = self.hugeThreshSize
            C = np.std(image)/np.mean(image)
        thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, ksize, C)
        # value, thresh = cv.threshold(image, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)  # TODO
        # print(value)
        # thresh[thresh < 0] = 0

        return thresh

    def gaussionFilter(self, image, ishuge, proportion=0.03):
        # Gauss filtering
        # [height, width, channels] = image.shape
        height = image.shape[0]
        size = int(height * self.normalBlurSize) + (int(height * self.normalBlurSize) - 1) % 2
        if ishuge:
            size = int(height * proportion) + (int(height * proportion) - 1) % 2
        blur = cv.GaussianBlur(image, (size, size), 0)

        return blur

    def biFilter(self, image, ishuge):
        # bilateralFilter
        blur = cv.bilateralFilter(image, 0, 10, 5)

        return blur

    def medianFilter(selfs, image, ishuge):
        # MedianFilter
        [height, width] = image.shape
        size = int(height * 0.03) + (int(height * 0.03) - 1) % 2
        if ishuge:
            size = int(height * 0.035) + (int(height * 0.035) - 1) % 2

        blur = cv.medianBlur(image, size)

        return blur

    def drawRectangle(self, image, ratio, baseImage, name, ishuge=False):

        [height, width] = image.shape

        cnts = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]
        thickness = 3 - round(len(cnts) / 100)
        if ishuge:
            thickness = 4 - round(len(cnts) / 50)   # 5 100 TODO 60
        res = []
        print("     Number of contours: ", len(cnts))
        littleDiff = False
        if len(cnts) <= 5:
            littleDiff = True

        # The first iteration
        blank = np.zeros([int(height), int(width), 3], np.uint8)
        for c in cnts:
            x, y, w, h = cv.boundingRect(c)
            # print(x, y, w, h)
            if w < 0.9 * width and h < 0.9 * height:
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                cv.rectangle(blank, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=thickness)
        if self.showImage:
            cv.imshow('blank1', blank)

        # The second iteration
        # if ishuge:
        blank2 = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
        blank = np.zeros([int(height), int(width), 3], np.uint8)
        cnts = cv.findContours(blank2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]
        # i = 0
        for c in cnts:
            x, y, w, h = cv.boundingRect(c)
            # print(x, y, w, h)
            center_x, center_y = x + w // 2, y + h // 2
            if littleDiff or w > self.rectEdgeThresh * width and h > self.rectEdgeThresh * height:
                # if not (ishuge and not (
                #         (0.1 * width < center_x < 0.90 * width) and (0.1 * height < center_y < 0.90 * height))):
                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h
                    cv.rectangle(blank, (x1, y1), (x2, y2), (0, 0, 255), -1)
        if self.showImage:
            cv.imshow('blank2', blank)

        # if len(cnts) == 0 and not self.flag:
        #     self.flag = True
        #     tempFirstBlurSize = self.firstBlurSize
        #     tempSecondBlurSize = self.secondBlurSize
        #     # tempNormalBlurSize = self.normalBlurSize
        #     self.firstBlurSize = 0.012
        #     self.secondBlurSize = 0.012
        #     # self.normalBlurSize = 0.01
        #     res = self.onceDetection(self.originBaseImage, self.originTargetImage, name)
        #     self.firstBlurSize = tempFirstBlurSize
        #     self.secondBlurSize = tempSecondBlurSize
        #     self.flag = False
        #
        #     return res

        if ishuge:
            blank2 = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
            blank = np.zeros([int(height), int(width), 3], np.uint8)
            cnts = cv.findContours(blank2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnts = cnts[1] if imutils.is_cv3() else cnts[0]
            # i = 0
            for c in cnts:
                x, y, w, h = cv.boundingRect(c)
                # print(x, y, w, h)
                center_x, center_y = x + w // 2, y + h // 2
                if littleDiff or 0.05 * width < w < 0.95 * width and 0.05 * height < h < 0.95 * height or \
                        ((ishuge and (
                                width * 0.4 < center_x < width * 0.6 and height * 0.4 < center_y < height * 0.6) and
                          w * h >= width * height * 0.0025)) or \
                        (not ishuge and (
                                width * 0.3 < center_x < width * 0.7 or height * 0.3 < center_y < height * 0.7) and
                         w * h >= width * height * 0.0025):
                    if not (ishuge and not (
                            (0.1 * width < center_x < 0.91 * width) and (0.1 * height < center_y < 0.9 * height))):
                        x1 = x
                        y1 = y
                        x2 = x + w
                        y2 = y + h
                        cv.rectangle(blank, (x1, y1), (x2, y2), (0, 0, 255), -1)
            if self.showImage:
                cv.imshow('blank2.5', blank)

        # The third iteration
        blank2 = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
        cnts = cv.findContours(blank2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]
        for c in cnts:
            x, y, w, h = cv.boundingRect(c)
            center_x, center_y = x + w // 2, y + h // 2
            if littleDiff or 0.05 * width < w < 0.95 * width and 0.05 * height < h < 0.95 * height or \
                    ((ishuge and (width * 0.4 < center_x < width * 0.6 and height * 0.4 < center_y < height * 0.6) and
                     w * h >= width * height * 0.0025)) or \
                    (not ishuge and (width * 0.3 < center_x < width * 0.7 or height * 0.3 < center_y < height * 0.7) and
                     w * h >= width * height * 0.0025):
                if not (ishuge and not (
                         (0.09 * width < center_x < 0.91 * width) and (0.15 * height < center_y < 0.91 * height))):
                    x1 = int(x * ratio[1])
                    y1 = int(y * ratio[0])
                    x2 = int((x + w) * ratio[1])
                    y2 = int((y + h) * ratio[0])
                    cv.rectangle(baseImage, (x, y), ((x + w), (y + h)), color=(0, 0, 255), thickness=0)
                    print("     ", x1, y1, x2, y2)
                    item = [str(x1), str(y1), str(x2), str(y2)]
                    res.append(item)

        cv.imwrite("{}.png".format("./images_res/" + name), baseImage)
        if self.showImage:
            cv.imshow('blank3', baseImage)

        return res


if __name__ == "__main__":
    # dataset = ImageDataSet("./test1/")
    dataset = ImageDataSet("./test2")
    # dataset = ImageDataSet("./try/")
    # dataset = ImageDataSet("./small/")
    model = DetectionModel(0.6, 0.6, showImage=False, pause=False, nFeatures=2000, hugeImageThresh=200000,  # TODO
                           normalBlurSize=0.014,
                           firstBlurSize=0.035, secondBlurSize=0.015,   # 25 15
                           normalThreshSize=3, normalThreshC=2.5,
                           hugeThreshSize=7, hugeThreshC=4.5     # 9 4.5   TODO:5 2.5
                           )
    model.detection(dataset, './detections/')
    # model.detection(dataset, './try_res/')
    # model.detection(dataset, './small_res/')
    cv.waitKey(0)

# Qvief0d7wQkmHx7f3fpkk14FUz1VlxHvcRhEtzqPPQ1pWlvyOMyyY8TeNCLw9jH3 80.17 points
# HI4LFtpYM6CCtkOPVulFDq+xAXPltQ4J3v0fNll3dDwZSyB209z3oym1/X0LEj1I 80.39 points (25 15 7 4.5 4/50)
# rOG757edPHpuyp2F+tiQ160ML8VrZHnO+tflwcaFhQst8bmQ6ti2carXFipwTF6r 80.29 points
# lYfgrigPfpH7qB9q8S8Q6kzUmlSad5S2qk83bhdn/ZQwMEDAyCtzn42TDwzp7s98 80.23 points
# +XSCdDmFP5VEcpvoXpIWy3tpqmQNXMuMUKopoR/S5nZlm6ERzAqwQ99F2sSUW81W 80.14 points
# 11PY28L7oB00YhcLomxYzQSqMpiYJtHQyWjAZ5WvEBLyoL2Owfg+N4N616n7YYJ0 80.04 points
# sdem/N/7civVMHt0yPf1Bm+8BSUx7UZUuIRKdNIUbJjjLD3G/Q0qMflquTOjXhns 79.94 points
# vFVEPQNeGfmGY2Ax3ncrosuWj4vVYqjhR28pyywx+Phab8C1Jy4y4RNCxTwx4AlZ 79.86 points
# WxVrg6so9R/jeypq9y/SQsQHcNEc/n+bn1YR6MR9xuGRX/zx0K8ZRae5py/LMpke 79.69 points
# DQMQJkj4IbyEtq/b6boPtkmyk2HKiVcQiuKkvQS3DqD4lwAB69sSvezx+4Ey9Hp/ 79.59 points

