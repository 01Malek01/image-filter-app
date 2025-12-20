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
