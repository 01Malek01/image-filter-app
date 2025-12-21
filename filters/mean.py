"""Mean Filter (Averaging Blur)

Purpose:
    Applies a mean (averaging) filter to smooth the image by reducing noise and detail.
    Each pixel is replaced by the average of all pixels in a neighborhood around it.
    This is a simple and effective method for noise reduction, though it can blur edges.

Steps:
    1. Convert the input BGR image to grayscale
    2. Apply the mean filter using cv2.blur() with a 3x3 kernel
    3. Each output pixel becomes the average of all pixels in the 3x3 neighborhood
    4. Convert the smoothed result back to BGR color space for display consistency
"""

import cv2


def apply_mean(image):
    """Apply mean filter (blur)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_filtered = cv2.blur(gray, (3, 3))
    return cv2.cvtColor(mean_filtered, cv2.COLOR_GRAY2BGR)
