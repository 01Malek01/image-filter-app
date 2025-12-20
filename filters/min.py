import cv2
import numpy as np


def apply_min(image):
    """Apply min filter (erosion)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    min_filtered = cv2.erode(gray, kernel)
    return cv2.cvtColor(min_filtered, cv2.COLOR_GRAY2BGR)
