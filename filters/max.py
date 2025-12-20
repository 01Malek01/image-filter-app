import cv2
import numpy as np


def apply_max(image):
    """Apply max filter (dilation)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    max_filtered = cv2.dilate(gray, kernel)
    return cv2.cvtColor(max_filtered, cv2.COLOR_GRAY2BGR)
