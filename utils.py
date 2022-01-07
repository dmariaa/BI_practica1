import cv2
import cv2.cv2 as cv2
import numpy as np


def generate_bounding_box(mask, opening_iterations=4, opening_kernel_size=(3, 3)):
    kernel = np.ones(opening_kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, iterations=opening_iterations, kernel=kernel)
    rect = cv2.boundingRect(mask)
    return rect
