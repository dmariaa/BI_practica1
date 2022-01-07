import glob

import cv2.cv2 as cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt

ear_cascade = cv2.CascadeClassifier('haar_files/cascade.xml')


def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = gray * mask
    boxes = ear_cascade.detectMultiScale(gray,
                                         scaleFactor=1.05,
                                         minNeighbors=5,
                                         minSize=(100, 100),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    # boxes, weights = cv2.groupRectangles(boxes.tolist(), 1, 0.95)
    boxes = non_max_suppression(boxes, overlapThresh=0.1)

    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

    return img


for file in sorted(glob.glob("data/test/*.bmp")):
    img = cv2.imread(file)
    img = detect(img)

    plt.imshow(img)
    plt.show()
