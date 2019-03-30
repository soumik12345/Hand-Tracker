import numpy as np
import cv2

class Face_Remover:

    def __init__(self, cascade_location = 'haarcascade_frontalface_default.xml'):
        self.face_cascade = cv2.CascadeClassifier(cascade_location)
    
    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # roi_gray = gray[y:y+h, x:x+w]
            frame[y : y + h, x : x + w] = np.zeros_like(frame[y : y + h, x : x + w])
        return frame