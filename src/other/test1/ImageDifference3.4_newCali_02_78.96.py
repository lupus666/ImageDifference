import cv2 as cv
import imutils
import numpy as np
import os
import time


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

    def sift_kp(self, image):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 颜色空间转换
        sift = cv.xfeatures2d.SIFT_create(self.nFeature)
        # sift = cv.AKAZE_create()
        kps, des = sift.detectAndCompute(image, None)
        kp_image = cv.drawKeypoints(gray_image, kps, None)  # 绘制关键点的函数
        return kp_image, kps, des

    def get_good_match(self, des1, des2):
        matcher = cv.FlannBasedMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        # for i in range(len(matches) - 1):
        #     if matches[i].distance < 0.99 * matches[i + 1].distance:
        #         good.append(matches[i])
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)
        return good

    def siftImageAlignment(self, img1, img2):
        _, kp1, des1 = self.sift_kp(img1)
        _, kp2, des2 = self.sift_kp(img2)
        goodMatch = self.get_good_match(des1, des2)
        imgOut = img2
        H = 0
        status = 0
        if len(goodMatch) > 4:
            ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ransacReprojThreshold = 2.0
            H, status = cv.findHomography(ptsA, ptsB, cv.RANSAC, ransacReprojThreshold)
            imgOut = cv.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]),
                                        flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_REPLICATE)
            return imgOut, H, status
        else:
            return imgOut, H, status


class DetectionModel(object):
    def __init__(self, edgeX=1.0, edgeY=1.0, threshold=0, morphologyIter=1, rectEdgeThresh=0.03, calibrate=True,
                 nFeatures=2000, showImage=False, pause=False):
        self.threshold = threshold
        self.edgeX = edgeX
        self.edgeY = edgeY
        self.morphologyIter = morphologyIter
        self.rectEdgeThresh = rectEdgeThresh
        self.calibrate = calibrate
        self.nFeatures = nFeatures
        self.showImage = showImage
        self.pause = pause

    def detection(self, dataset, respath):
        resize_time = 0
        callable_time = 0
        diff_time = 0
        deEdge_time = 0
        morph_time = 0
        filter_time = 0
        drawRect_time = 0
        gaussian_time = 0
        for i in range(0, len(dataset)):
            print("{}/{}:".format(i + 1, len(dataset)))

            ishuge = False
            name, baseImage, targetImage = dataset[i]
            # if self.showImage:
            #     cv.imshow('origin1', baseImage)
            #     cv.imshow('origin2', targetImage)
            height, width, channels = baseImage.shape

            resize_height = 320
            resize_width = 320
            if height * width > 360000:
                ishuge = True
                resize_height = 320
                resize_width = 320
                self.morphologyIter = 1

            # Calibration
            callable_start_time = time.time()
            if self.calibrate:
                targetImage = self.doCalibrate(baseImage, targetImage)
            callable_end_time = time.time()
            callable_time += callable_end_time - callable_start_time
            # if self.showImage:
            #     cv.imshow('calibrate', targetImage)

            # DeEdge
            if ishuge:
                deEdge_start_time = time.time()
                baseImage = self.deEdge(baseImage)
                targetImage = self.deEdge(targetImage)
                deEdge_end_time = time.time()
                deEdge_time += deEdge_end_time - deEdge_start_time

            # Gaussian filtering
            if ishuge:
                gaussian_start_time = time.time()
                baseImage = self.gaussionFilter(baseImage, ishuge)
                targetImage = self.gaussionFilter(targetImage, ishuge)
                gaussian_end_time = time.time()
                gaussian_time += gaussian_end_time - gaussian_start_time

            # Resize
            resize_start_time = time.time()
            baseImage, height_ratio1, width_ratio1 = self.resize(image=baseImage, dsize=(resize_width, resize_height))
            targetImage, height_ratio2, width_ratio2 = self.resize(image=targetImage,
                                                                   dsize=(resize_width, resize_height))
            resize_end_time = time.time()
            resize_time += resize_end_time - resize_start_time
            if self.showImage:
                cv.imshow('resize1', baseImage)
                cv.imshow('resize2', targetImage)

            # DeEdge
            # if ishuge:
            #     deEdge_start_time = time.time()
            #     baseImage = self.deEdge(baseImage)
            #     targetImage = self.deEdge(targetImage)
            #     deEdge_end_time = time.time()
            #     deEdge_time += deEdge_end_time - deEdge_start_time
            # baseImage = resizebaseImage
            # targetImage = resizeTargetImage

            # Morphology operation
            morphology_start_time = time.time()
            baseImage = self.morphology(baseImage)
            targetImage = self.morphology(targetImage)
            morphology_end_time = time.time()
            morph_time += morphology_end_time - morphology_start_time
            # if self.showImage:
            #     cv.imshow('morphology1', baseImage)
            #     cv.imshow('morphology2', targetImage)

            # Gaussian filtering
            gaussian_start_time = time.time()
            baseImage = self.gaussionFilter(baseImage, ishuge)
            targetImage = self.gaussionFilter(targetImage, ishuge)
            gaussian_end_time = time.time()
            gaussian_time += gaussian_end_time - gaussian_start_time
            # if self.showImage:
            #     cv.imshow('filter1', baseImage)
            #     cv.imshow('filter2', targetImage)

            # Color Difference
            diff_start_time = time.time()
            diff = self.colorDifference(baseImage, targetImage).astype(np.uint8)
            diff_end_time = time.time()
            diff_time += diff_end_time - diff_start_time
            if self.showImage:
                cv.imshow('diff', diff)

            # gaussian_start_time = time.time()
            # diff = self.medianFilter(diff, ishuge)
            # # targetImage = self.gaussionFilter(targetImage, ishuge)
            # gaussian_end_time = time.time()
            # gaussian_time += gaussian_end_time - gaussian_start_time

            # Thresh filtering
            filter_start_time = time.time()
            thresh = self.thresholdFilter(diff)
            filter_end_time = time.time()
            filter_time += filter_end_time - filter_start_time
            if self.showImage:
                cv.imshow('thresh', thresh)

            # Draw rectangle
            drawRectangle_start_time = time.time()
            items = self.drawRectangle(thresh, (height_ratio1, width_ratio1), baseImage, ishuge)
            drawRectangle_end_time = time.time()
            drawRect_time += drawRectangle_end_time - drawRectangle_start_time

            # Output
            f = open(os.path.join(respath, name + '.txt'), 'w')
            for item in items:
                strs = " ".join(item)
                strs = "True 1.0 " + strs + '\n'
                f.write(strs)

            f.close()

            if self.pause:
                cv.waitKey(3000)

        total_time = callable_time + diff_time + deEdge_time + morph_time + filter_time + drawRect_time
        print("Total time: {}".format(total_time))
        print("Calibration time: {}".format(callable_time))
        print("Diff time: {}".format(diff_time))
        print("DeEdge time: {}".format(deEdge_time))
        print("Morph time: {}".format(morph_time))
        print("Filter time: {}".format(filter_time))
        print("DrawRect time: {}".format(drawRect_time))

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
        kernel = np.array([[0, -1, 0], [-1, 7, -1], [0, -1, 0]], np.float32)
        shapeImage = cv.filter2D(image, -1, kernel=kernel)
        return shapeImage

    def morphology(self, image):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        thresh = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=self.morphologyIter)

        return thresh

    def thresholdFilter(self, image):
        # Threshold filtering C could be adaptive
        thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 3, 2.5)
        # thresh[thresh < 0] = 0

        return thresh

    def gaussionFilter(self, image, ishuge):
        # Gauss filtering
        [height, width, channels] = image.shape
        size = int(height * 0.014) + (int(height * 0.014) - 1) % 2
        if ishuge:
            size = int(height * 0.03) + (int(height * 0.03) - 1) % 2
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

    def drawRectangle(self, image, ratio, baseImage, ishuge=False):

        [height, width] = image.shape

        cnts = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]
        thickness = 3 - round(len(cnts) / 100)
        if ishuge:
            thickness = 4 - round(len(cnts) / 80)
        res = []
        print(len(cnts))
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
                if not (ishuge and not (
                        (0.1 * width < center_x < 0.9 * width) and (0.1 * height < center_y < 0.9 * height))):
                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h
                    cv.rectangle(blank, (x1, y1), (x2, y2), (0, 0, 255), -1)
        if self.showImage:
            cv.imshow('blank2', blank)

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
                if littleDiff or w > 0.05 * width and h > 0.05 * height:
                    if not (ishuge and not (
                            (0.1 * width < center_x < 0.9 * width) and (0.1 * height < center_y < 0.9 * height))):
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
                    (ishuge and (width * 0.4 < center_x < width * 0.6 and height * 0.4 < center_y < height * 0.6) and
                     w * h >= width * height * 0.0025) or \
                    (not ishuge and (width * 0.4 < center_x < width * 0.6 or height * 0.4 < center_y < height * 0.6) and
                     w * h >= width * height * 0.0032):
                if not (ishuge and not (
                         (0.1 * width < center_x < 0.9 * width) and (0.1 * height < center_y < 0.9 * height))):
                    x1 = int(x * ratio[1])
                    y1 = int(y * ratio[0])
                    x2 = int((x + w) * ratio[1])
                    y2 = int((y + h) * ratio[0])
                    cv.rectangle(baseImage, (x, y), ((x + w), (y + h)), color=(0, 0, 255), thickness=0)
                    print("     ", x1, y1, x2, y2)
                    item = [str(x1), str(y1), str(x2), str(y2)]
                    res.append(item)
        if self.showImage:
            cv.imshow('blank3', baseImage)

        return res


if __name__ == "__main__":
    dataset = ImageDataSet("./test2/")
    model = DetectionModel(0.6, 0.6, showImage=False, pause=False)
    model.detection(dataset, './detections/')
    # cv.waitKey(0)

# oOa86fXBFAAr7vNFIpcD3dzFc9dC0/d8cPj6W31SurYuC5dPraOTS6n1QrfdFbjk 78.69 points
# J6xIJhKCUD/P+KVprfGVl4gzym3Uf0baQHf5QdaoLufde3QyeyVsVVMZzkJjWjyj 78.05 points
# ae4rH/c+3JKYsGS6wxDz9xrFu54ltp3lfr0rGMVjUIbQ1p6SX96hw6Ozn/SSV43h 78.13 points
# hAliQRo/mp6eRjSEx5Lb+tmoL6idWZIsl7Gxci95pzj3uS6UbdzyAE/JpG269+vl 77.50 points
# BXDzbixN60mttfgVTVoPyUXmHBg5VU7Fid+ThefQEJVE3W1KUwJBtRcqMDhFRZG6 77.41 points
# L+mjx/MfslK0aDkHkvdp3/Fsgcd/kFm+E26h9YOJyDUqEkdAgL0w7ByP6jyzeuEc 77.14 points
# VymrjL+P4ASsriklL7PeoBlFxvQT23KwcDce2ukxD4XVQ4nPayiEtBfvKoSyrKvi 77.06 points
# o4AyJcFUcWP9MoOOiELR5DROc62ZViscpC3na/pI0NhvSzMOF+rW+YtxzS3K5pNC 76.74 points
# HxLXIqRgjn3RVUggRpQlU5XixUjE2/p9aYZ/yCnuFDVntsEja2wCqKiqcvd/k/mb 76.81 points
# penkTOPgX3IzZIx6IRjZs7n+xY64SK6eqOIs90haEfpYajzPnK7yEbzT7MYOhdpb 76.83 points
# 0VQuJejUGdlWJUez6AXk5vHW4rUnivFnMCQxzFXwScAK26QzfDtTyNs4PDbqNpDT 76.98 points

