"""Min Filter (Erosion)

Purpose:
    Applies a minimum filter to the image, which is morphologically equivalent to
    erosion. This filter replaces each pixel with the minimum value in its neighborhood,
    making dark regions expand. It's useful for removing small bright noise (salt noise),
    separating connected objects, and enhancing dark features.

Steps:
    1. Convert the input BGR image to grayscale
    2. Create a 3x3 kernel of ones for the morphological operation
    3. Apply erosion using cv2.erode() which selects the minimum pixel value in the kernel area
    4. Convert the result back to BGR color space for display consistency
"""

import cv2
import numpy as np


def apply_min(image):
    """Apply min filter (erosion)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    min_filtered = cv2.erode(gray, kernel)
    return cv2.cvtColor(min_filtered, cv2.COLOR_GRAY2BGR)
