import cv2
import numpy as np


def apply_sobelx(image):
    """Apply Sobel X (horizontal edge detection)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sx = np.uint8(np.absolute(sx))
    return cv2.cvtColor(sx, cv2.COLOR_GRAY2BGR)


def apply_sobely(image):
    """Apply Sobel Y (vertical edge detection)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sy = np.uint8(np.absolute(sy))
    return cv2.cvtColor(sy, cv2.COLOR_GRAY2BGR)


def apply_sobel(image):
    """Apply combined Sobel (both X and Y)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sx = np.uint8(np.absolute(sx))
    sy = np.uint8(np.absolute(sy))
    combined = cv2.bitwise_or(sx, sy)
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
