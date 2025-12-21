"""Prewitt Edge Detection Filters

Purpose:
    Detects edges in an image using the Prewitt operator, a first-order derivative method
    that uses discrete differentiation to calculate the gradient of image intensity.
    The Prewitt filter is effective at detecting edges and emphasizing regions of high
    spatial frequency. It provides separate horizontal (X) and vertical (Y) edge detection,
    as well as a combined approach.

Steps (Common to all Prewitt functions):
    1. Convert the input BGR image to grayscale
    2. Define the appropriate Prewitt kernel(s):
       - Prewitt X: [[-1,0,1], [-1,0,1], [-1,0,1]] for horizontal edges
       - Prewitt Y: [[-1,-1,-1], [0,0,0], [1,1,1]] for vertical edges
    3. Apply the kernel using cv2.filter2D() to compute the gradient
    4. Take the absolute value to get positive edge magnitudes
    5. Convert to 8-bit unsigned integer format
    6. For combined Prewitt: merge X and Y results using bitwise OR
    7. Convert back to BGR color space for display consistency
"""

import cv2
import numpy as np


def apply_prewittx(image):
    """Apply Prewitt X (horizontal edge detection)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Prewitt X kernel
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    px = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    px = np.uint8(np.absolute(px))
    return cv2.cvtColor(px, cv2.COLOR_GRAY2BGR)


def apply_prewitty(image):
    """Apply Prewitt Y (vertical edge detection)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Prewitt Y kernel
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    py = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
    py = np.uint8(np.absolute(py))
    return cv2.cvtColor(py, cv2.COLOR_GRAY2BGR)


def apply_prewitt(image):
    """Apply combined Prewitt (both X and Y)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Prewitt X and Y kernels
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    px = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    py = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
    px = np.uint8(np.absolute(px))
    py = np.uint8(np.absolute(py))
    combined = cv2.bitwise_or(px, py)
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
