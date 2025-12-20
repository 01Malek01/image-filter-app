import cv2


def apply_median(image):
    """Apply median filter."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_filtered = cv2.medianBlur(gray, 3)
    return cv2.cvtColor(median_filtered, cv2.COLOR_GRAY2BGR)
