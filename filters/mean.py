import cv2


def apply_mean(image):
    """Apply mean filter (blur)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_filtered = cv2.blur(gray, (3, 3))
    return cv2.cvtColor(mean_filtered, cv2.COLOR_GRAY2BGR)
