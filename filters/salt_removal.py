import cv2
import numpy as np


def apply_salt_removal(image):
    """Apply salt noise removal using aggressive dilation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    salt_removed = cv2.dilate(gray, kernel, iterations=2)
    return cv2.cvtColor(salt_removed, cv2.COLOR_GRAY2BGR)
