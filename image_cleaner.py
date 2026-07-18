"""
Image preprocessing pipeline to improve Tesseract OCR accuracy.

Operations applied in order:
  1. Grayscale conversion     — reduces noise, Tesseract works on grayscale
  2. Denoising                — removes scanner artifacts and compression noise
  3. Deskewing                — corrects page tilt (common in scanned docs)
  4. Thresholding (binarize)  — produces clean black-on-white for Tesseract

Each step is independently togglable for debugging purposes.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def clean_image(
    image: Image.Image,
    denoise: bool = True,
    deskew: bool = True,
    threshold: bool = True,
) -> Image.Image:
    """
    Apply preprocessing pipeline to a PIL image before OCR.

    Args:
        image: Input PIL image (RGB or grayscale).
        denoise: Apply fast non-local means denoising.
        deskew: Detect and correct page skew.
        threshold: Apply Otsu's binarization.

    Returns:
        Cleaned PIL image ready for Tesseract.
    """
    img = _to_cv2_gray(image)

    if denoise:
        img = _denoise(img)

    if deskew:
        img = _deskew(img)

    if threshold:
        img = _threshold(img)

    return _to_pil(img)


# ---------------------------------------------------------------------------
# Internal steps
# ---------------------------------------------------------------------------

def _to_cv2_gray(image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV grayscale numpy array."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _denoise(img: np.ndarray) -> np.ndarray:
    """
    Fast non-local means denoising.
    h=10 is a reasonable default — higher removes more noise but blurs edges.
    For heavily degraded scans, increase to 15-20.
    """
    return cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)


def _deskew(img: np.ndarray) -> np.ndarray:
    """
    Detect and correct page skew by finding the dominant text angle.

    Strategy:
    - Threshold the image temporarily to find text blobs
    - Use minAreaRect on all non-zero points to estimate skew angle
    - Rotate the original image to correct the skew

    Limitations:
    - Works well for skew < ~10 degrees (typical scanner output)
    - Will fail on severely rotated pages (90°, 180°) — handle those separately
    - Noisy images with few text regions may produce inaccurate angle estimates
    """
    # Binary threshold to isolate text regions
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))

    if len(coords) < 50:
        # Not enough content to estimate skew reliably
        logger.debug("Insufficient content for deskew, skipping")
        return img

    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect returns angles in [-90, 0); normalize to [-45, 45)
    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5:
        # Skew too small to matter — skip rotation to avoid interpolation artifacts
        return img

    logger.debug("Deskewing by %.2f degrees", angle)

    h, w = img.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        img,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _threshold(img: np.ndarray) -> np.ndarray:
    """
    Otsu's binarization — automatically determines the optimal threshold.
    Produces clean black text on white background for Tesseract.

    For documents with uneven lighting (e.g., photographed pages),
    adaptive thresholding would be better:
        cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 31, 10)
    Switch to adaptive if Otsu gives poor results on your target documents.
    """
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV grayscale array back to PIL image."""
    return Image.fromarray(img)
