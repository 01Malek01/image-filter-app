import cv2
import numpy as np


def apply_laplacian(image):
    """Apply Laplacian edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
