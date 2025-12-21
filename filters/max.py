"""Max Filter (Dilation)

Purpose:
    Applies a maximum filter to the image, which is morphologically equivalent to
    dilation. This filter replaces each pixel with the maximum value in its neighborhood,
    making bright regions expand. It's useful for removing small dark noise (pepper noise),
    connecting broken lines, and enhancing bright features.

Steps:
    1. Convert the input BGR image to grayscale
    2. Create a 3x3 kernel of ones for the morphological operation
    3. Apply dilation using cv2.dilate() which selects the maximum pixel value in the kernel area
    4. Convert the result back to BGR color space for display consistency
"""

import cv2
import numpy as np


def apply_max(image):
    """Apply max filter (dilation)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    max_filtered = cv2.dilate(gray, kernel)
    return cv2.cvtColor(max_filtered, cv2.COLOR_GRAY2BGR)
