import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True:
    flag, img = cap.read()
    cv2.imshow('name', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()