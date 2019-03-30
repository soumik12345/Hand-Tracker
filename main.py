import cv2
import numpy as np
from utils.Face_Remover import *

face_remover = Face_Remover('./front_face_cascade.xml')

video_capture = cv2.VideoCapture(0)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, frame = video_capture.read()
    removed_face = face_remover.detect(frame)
    cv2.imshow("Video", removed_face)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()