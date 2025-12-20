from .mean import apply_mean
from .median import apply_median
from .min import apply_min
from .pepper_removal import apply_pepper_removal
from .max import apply_max
from .salt_removal import apply_salt_removal
from .sobel import apply_sobelx, apply_sobely, apply_sobel
from .laplacian import apply_laplacian
from .prewitt import apply_prewittx, apply_prewitty, apply_prewitt

__all__ = [
    "apply_mean",
    "apply_median",
    "apply_min",
    "apply_pepper_removal",
    "apply_max",
    "apply_salt_removal",
    "apply_sobelx",
    "apply_sobely",
    "apply_sobel",
    "apply_laplacian",
    "apply_prewittx",
    "apply_prewitty",
    "apply_prewitt",
]
