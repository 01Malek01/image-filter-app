"""Pepper Noise Removal Filter

Purpose:
    Removes pepper noise (small dark spots) from images using morphological erosion.
    Pepper noise appears as isolated black pixels scattered throughout the image.
    This filter specifically targets and reduces these dark artifacts while preserving
    the overall image structure.

Steps:
    1. Convert the input BGR image to grayscale
    2. Create a 3x3 kernel of ones for the morphological operation
    3. Apply erosion using cv2.erode() which removes small dark spots by selecting minimum values
    4. The erosion operation makes bright regions shrink and eliminates isolated dark pixels
    5. Convert the result back to BGR color space for display consistency
"""

import cv2
import numpy as np


def apply_pepper_removal(image):
    """Apply pepper noise removal using erosion."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    pepper_removed = cv2.erode(gray, kernel)
    return cv2.cvtColor(pepper_removed, cv2.COLOR_GRAY2BGR)
