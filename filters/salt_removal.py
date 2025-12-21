"""Salt Noise Removal Filter

Purpose:
    Removes salt noise (small bright spots) from images using aggressive morphological
    dilation. Salt noise appears as isolated white pixels scattered throughout the image.
    This filter specifically targets and reduces these bright artifacts while preserving
    the overall image structure.

Steps:
    1. Convert the input BGR image to grayscale
    2. Create a larger 5x5 kernel of ones for more aggressive morphological operation
    3. Apply dilation using cv2.dilate() with 2 iterations
    4. The dilation operation makes dark regions expand, filling in isolated bright pixels
    5. Multiple iterations ensure more effective removal of salt noise
    6. Convert the result back to BGR color space for display consistency
"""

import cv2
import numpy as np


def apply_salt_removal(image):
    """Apply salt noise removal using aggressive dilation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    salt_removed = cv2.dilate(gray, kernel, iterations=2)
    return cv2.cvtColor(salt_removed, cv2.COLOR_GRAY2BGR)
