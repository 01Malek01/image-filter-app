"""Sobel Edge Detection Filters

Purpose:
    Detects edges in an image using the Sobel operator, a first-order derivative method
    that combines Gaussian smoothing with differentiation for better noise resistance.
    The Sobel filter calculates the gradient of image intensity at each pixel, providing
    information about edge direction and magnitude. It offers separate horizontal (X) and
    vertical (Y) edge detection, as well as a combined approach.

Steps (Common to all Sobel functions):
    1. Convert the input BGR image to grayscale
    2. Apply the Sobel operator using cv2.Sobel():
       - Sobel X: derivatives in X direction (dx=1, dy=0) for horizontal edges
       - Sobel Y: derivatives in Y direction (dx=0, dy=1) for vertical edges
    3. Use CV_64F data type to handle negative values from the derivative
    4. Take the absolute value to get positive edge magnitudes
    5. Convert to 8-bit unsigned integer format
    6. For combined Sobel: merge X and Y results using bitwise OR for complete edge map
    7. Convert back to BGR color space for display consistency
"""

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
