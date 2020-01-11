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
        sift = cv.xfeatures2d_SIFT.create(self.nFeature)
        kps, des = sift.detectAndCompute(image, None)
        kp_image = cv.drawKeypoints(gray_image, kps, None)  # 绘制关键点的函数
        return kp_image, kps, des

    def get_good_match(self, des1, des2):
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.99 * n.distance:
                good.append(m)
        return good

    def siftImageAlignment(self, img1, img2):
        _, kp1, des1 = self.sift_kp(img1)
        _, kp2, des2 = self.sift_kp(img2)
        goodMatch = self.get_good_match(des1, des2)
        if len(goodMatch) > 4:
            ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ransacReprojThreshold = 2
            H, status = cv.findHomography(ptsA, ptsB, cv.RANSAC, ransacReprojThreshold)
            imgOut = cv.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]),
                                        flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_REPLICATE)
        return imgOut, H, status


class DetectionModel(object):
    def __init__(self, edgeX=1.0, edgeY=1.0, threshold=0, morphologyIter=1, rectEdgeThresh=0.03, calibrate=True,
                 nFeatures=2000, showImage=False, isPause=False):
        self.threshold = threshold
        self.edgeX = edgeX
        self.edgeY = edgeY
        self.morphologyIter = morphologyIter
        self.rectEdgeThresh = rectEdgeThresh
        self.calibrate = calibrate
        self.nFeatures = nFeatures
        self.showImage = showImage
        self.isPause = isPause

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
            if self.isPause:
                # os.system("pause")
                cv.waitKeyEx(10000)
                # time.sleep(10)

            name, baseImage, targetImage = dataset[i]
            if self.showImage:
                cv.imshow('origin1', baseImage)
                cv.imshow('origin2', targetImage)
            threshBaseImage = baseImage
            threshTargetImage = targetImage

            # Morphology operation
            morphology_start_time = time.time()
            threshBaseImage = self.morphology(threshBaseImage)
            threshTargetImage = self.morphology(threshTargetImage)
            morphology_end_time = time.time()
            morph_time += morphology_end_time - morphology_start_time
            if self.showImage:
                cv.imshow('morphology1', threshBaseImage)
                cv.imshow('morphology2', threshTargetImage)

            # Gaussian filtering
            gaussian_start_time = time.time()
            threshBaseImage = self.gaussionFilter(threshBaseImage)
            threshTargetImage = self.gaussionFilter(threshTargetImage)
            gaussian_end_time = time.time()
            gaussian_time += gaussian_end_time - gaussian_start_time
            if self.showImage:
                cv.imshow('filter1', threshBaseImage)
                cv.imshow('filter2', threshTargetImage)


            # Calibration
            callable_start_time = time.time()
            if self.calibrate:
                threshTargetImage = self.doCalibrate(threshBaseImage, threshTargetImage)
            callable_end_time = time.time()
            callable_time += callable_end_time - callable_start_time
            if self.showImage:
                cv.imshow('calibrate', threshTargetImage)

            # Resize
            height, width, channels = threshBaseImage.shape
            # resize_height, resize_width = height, width
            # if height * width > 1000000:
            resize_height = 320
            resize_width = 320
            resize_start_time = time.time()
            resizeBaseImage, height_ratio1, width_ratio1 = self.resize(image=threshBaseImage, dsize=(resize_width, resize_height))
            resizeTargetImage, height_ratio2, width_ratio2 = self.resize(image=threshTargetImage, dsize=(resize_width, resize_height))
            resize_end_time = time.time()
            resize_time += resize_end_time - resize_start_time
            if self.showImage:
                cv.imshow('resize1', resizeBaseImage)
                cv.imshow('resize2', resizeTargetImage)

            # DeEdge
            # deEdge_start_time = time.time()
            # threshBaseImage = self.deEdge(resizeBaseImage)
            # threshTargetImage = self.deEdge(resizeTargetImage)
            # deEdge_end_time = time.time()
            # deEdge_time += deEdge_end_time - deEdge_start_time
            threshBaseImage = resizeBaseImage
            threshTargetImage = resizeTargetImage

            # # Gaussian filtering
            # threshBaseImage = cv.GaussianBlur(threshBaseImage, (3, 3), 0)
            # threshTargetImage = cv.GaussianBlur(threshTargetImage, (3, 3), 0)


            # Color Difference
            diff_start_time = time.time()
            diff = self.colorDifference(threshBaseImage, threshTargetImage).astype(np.uint8)
            diff_end_time = time.time()
            diff_time += diff_end_time - diff_start_time
            if self.showImage:
                cv.imshow('diff', diff)

            # Thresh filtering
            filter_start_time = time.time()
            thresh = self.thresholdFilter(diff)
            filter_end_time = time.time()
            filter_time += filter_end_time - filter_start_time
            if self.showImage:
                cv.imshow('thresh', thresh)

            # Draw rectangle
            drawRectangle_start_time = time.time()
            items = self.drawRectangle(thresh, (height_ratio1, width_ratio1), baseImage)
            drawRectangle_end_time = time.time()
            drawRect_time += drawRectangle_end_time - drawRectangle_start_time

            # Output
            f = open(os.path.join(respath, name + '.txt'), 'w')
            for item in items:
                strs = " ".join(item)
                strs = "True 1.0 " + strs + '\n'
                f.write(strs)

            f.close()

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
        height_ratio = height / dsize[0]
        width_ratio = width / dsize[1]
        return cv.resize(image, dsize), height_ratio, width_ratio

    def colorDifference(self, image1, image2):
        b1, g1, r1 = cv.split(image1)
        b2, g2, r2 = cv.split(image2)

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

    def morphology(self, image):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        thresh = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=self.morphologyIter)

        return thresh

    def thresholdFilter(self, image):
        # Threshold filtering C could be adaptive
        thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 3, 2.5)

        return thresh

    def gaussionFilter(self, image):
        # Gauss filtering
        [height, width, channels] = image.shape
        size = int(height * 0.015) + (int(height * 0.015) - 1) % 2    # 0.014 to 0.03
        blur = cv.GaussianBlur(image, (size, size), 0)

        return blur

    def drawRectangle(self, image, ratio, baseImage):

        [height, width] = image.shape

        cnts = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]

        thickness = 3 - len(cnts) // 100
        if thickness < 0:
            thickness = 0
        res = []

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
            if w > self.rectEdgeThresh * width and h > self.rectEdgeThresh * height:
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                cv.rectangle(blank, (x1, y1), (x2, y2), (0, 0, 255), 0)
        if self.showImage:
            cv.imshow('blank2', blank)

        # The third iteration
        blank2 = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
        blank = np.zeros([int(height), int(width), 3], np.uint8)
        cnts = cv.findContours(blank2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]
        for c in cnts:
            x, y, w, h = cv.boundingRect(c)
            if w > 0.06 * width and h > 0.06 * height:
                # or (width * 0.3 < x + w//2 < width * 0.7 and height * 0.3 < y + h//2 < height * 0.7 and w * h > 0.003*width*height):
                x1 = int(x * ratio[1])
                y1 = int(y * ratio[0])
                x2 = int((x + w) * ratio[1])
                y2 = int((y + h) * ratio[0])
                cv.rectangle(baseImage, (x1, y1), (x2, y2), (0, 0, 255), 0)
                print("     ",x1, y1, x2, y2)
                item = [str(x1), str(y1), str(x2), str(y2)]
                res.append(item)

        if self.showImage:
            cv.imshow('blank3', baseImage)
        return res


if __name__ == "__main__":
    dataset = ImageDataSet("./test2/")
    model = DetectionModel(0.2, 0.2, showImage=False, isPause=False)
    model.detection(dataset, './detections/')
    cv.waitKey(0)
