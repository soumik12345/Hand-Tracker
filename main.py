import cv2
import numpy as np
from utils.Face_Remover import *
from utils.Hand_Detector import *

face_remover = Face_Remover('./front_face_cascade.xml')
hand_detector = Hand_Detector(np.array([0, 133, 77], np.uint8), np.array([255, 173, 127], np.uint8))

video_capture = cv2.VideoCapture(0)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, frame = video_capture.read()
    removed_face = face_remover.detect(frame)
    detected_hand = hand_detector.detect(removed_face)
    cv2.imshow("Video", detected_hand)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()