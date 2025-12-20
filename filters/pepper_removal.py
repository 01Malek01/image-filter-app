import cv2
import numpy as np


def apply_pepper_removal(image):
    """Apply pepper noise removal using erosion."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    pepper_removed = cv2.erode(gray, kernel)
    return cv2.cvtColor(pepper_removed, cv2.COLOR_GRAY2BGR)
