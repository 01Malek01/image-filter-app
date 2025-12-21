"""Median Filter

Purpose:
    Applies a median filter to reduce noise while preserving edges better than
    averaging filters. Each pixel is replaced by the median value of all pixels in
    its neighborhood. This filter is particularly effective at removing salt-and-pepper
    noise while maintaining sharp edges.

Steps:
    1. Convert the input BGR image to grayscale
    2. Apply the median filter using cv2.medianBlur() with a 3x3 kernel
    3. Each output pixel becomes the median (middle value) of all pixels in the neighborhood
    4. Convert the filtered result back to BGR color space for display consistency
"""

import cv2


def apply_median(image):
    """Apply median filter."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_filtered = cv2.medianBlur(gray, 3)
    return cv2.cvtColor(median_filtered, cv2.COLOR_GRAY2BGR)
