"""Laplacian Filter

Purpose:
    Detects edges in an image using the Laplacian operator, which is a second-order
    derivative method that highlights regions of rapid intensity change (edges).
    The Laplacian filter is particularly good at finding edges without regard to
    their direction, making it useful for edge detection in various applications.

Steps:
    1. Convert the input BGR image to grayscale
    2. Apply the Laplacian operator using cv2.Laplacian() with CV_64F data type
    3. Take the absolute value to get positive edge magnitudes
    4. Convert the result to 8-bit unsigned integer format
    5. Convert back to BGR color space for display consistency
"""

import cv2
import numpy as np


def apply_laplacian(image):
    """Apply Laplacian edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
