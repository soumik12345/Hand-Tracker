import numpy as np
import cv2

class Hand_Detector:

    def __init__(self, min_YCrCb, max_YCrCb):
        self.min_YCrCb = min_YCrCb
        self.max_YCrCb = max_YCrCb
    
    def detect(self, image):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        skinRegion = cv2.inRange(ycrcb, self.min_YCrCb, self.max_YCrCb)
        _, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 1000:
                cv2.drawContours(image, contours, i, (0, 255, 0), 3)
        return image