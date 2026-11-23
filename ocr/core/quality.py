from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr.core.errors import ImageQualityError

GLARE_PIXEL_VALUE = 250


@dataclass(frozen=True)
class QualityThresholds:
    """Thresholds applied by the image quality gate.

    Attributes:
        min_sharpness: Minimum variance of the Laplacian of the grayscale image.
        min_contrast: Minimum standard deviation of grayscale pixel values.
        max_glare_ratio: Maximum share of near-saturated pixels.
    """

    min_sharpness: float = 50.0
    min_contrast: float = 20.0
    max_glare_ratio: float = 0.35


@dataclass(frozen=True)
class ImageQualityReport:
    """Measured quality metrics of a receipt image.

    Attributes:
        sharpness: Variance of the Laplacian of the grayscale image.
        contrast: Standard deviation of grayscale pixel values.
        glare_ratio: Share of near-saturated pixels in the image.
        accepted: Whether all metrics satisfy the configured thresholds.
    """

    sharpness: float
    contrast: float
    glare_ratio: float
    accepted: bool


def assess_quality(image: np.ndarray, thresholds: QualityThresholds) -> ImageQualityReport:
    """Measure sharpness, contrast and glare of an image.

    Args:
        image: BGR or grayscale image array.
        thresholds: Thresholds used to decide whether the image is acceptable.

    Returns:
        Measured metrics and the acceptance decision.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    glare_ratio = float(np.count_nonzero(gray >= GLARE_PIXEL_VALUE) / gray.size)
    accepted = (
        sharpness >= thresholds.min_sharpness
        and contrast >= thresholds.min_contrast
        and glare_ratio <= thresholds.max_glare_ratio
    )
    return ImageQualityReport(
        sharpness=sharpness,
        contrast=contrast,
        glare_ratio=glare_ratio,
        accepted=accepted,
    )


def ensure_quality(image: np.ndarray, thresholds: QualityThresholds) -> ImageQualityReport:
    """Reject an image that is too blurry, flat or glare-affected for OCR.

    Args:
        image: BGR or grayscale image array.
        thresholds: Thresholds used to decide whether the image is acceptable.

    Returns:
        Measured metrics when the image passes the gate.

    Raises:
        ImageQualityError: If any measured metric violates the thresholds.
    """
    report = assess_quality(image, thresholds)
    if report.accepted:
        return report
    raise ImageQualityError(
        "Image quality is too low: "
        f"sharpness={report.sharpness:.2f} (min {thresholds.min_sharpness}), "
        f"contrast={report.contrast:.2f} (min {thresholds.min_contrast}), "
        f"glare={report.glare_ratio:.3f} (max {thresholds.max_glare_ratio})"
    )
