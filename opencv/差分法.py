import cv2
import numpy as np


cap = cv2.VideoCapture(0)
flag, img1 = cap.read()
while True:
    flag, img2 = cap.read()
    diff = cv2.absdiff(img1, img2)
    cv2.imshow('name', diff)
    img1 = img2

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()