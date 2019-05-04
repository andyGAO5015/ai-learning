import cv2
import numpy as np


def get_mask(points, shape):
    """
    返回遮罩，在不规则多边形区域中为白色，外面为黑色
    :param points:
    :param shape:
    :return:
    """
    mask = np.zeros(shape, np.uint8)
    pts = points.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    return mask


def get_roi(img, points):
    """
    返回roi区域
    :param img:
    :param points:
    :return:
    """
    mask = get_mask(points, img.shape)
    roi = cv2.bitwise_and(img, mask)
    return roi


# 视频
cap = cv2.VideoCapture(0)
ret, image = cap.read()
shape = image.shape
points = np.array([[10, 10], [50, 500], [400, 400], [500, 10]], np.int32)
mask = get_mask(points, shape)
while ret:
    roi = cv2.bitwise_and(image, mask)
    #     roi = get_roi(image, points)
    cv2.imshow('name', roi)
    ret, image = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
