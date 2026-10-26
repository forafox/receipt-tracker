from __future__ import annotations

import cv2
import numpy as np

from ocr.core.errors import ImageDecodeError, ImageTooLargeError

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def decode_image(
    image: bytes,
    *,
    max_bytes: int = MAX_IMAGE_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> np.ndarray:
    """Decode raw image bytes into a BGR ``numpy`` array.

    Args:
        image: Encoded image bytes (JPEG, PNG, ...).
        max_bytes: Maximum accepted payload size in bytes.
        max_pixels: Maximum accepted decoded image area in pixels.

    Returns:
        Decoded BGR image array.

    Raises:
        ImageTooLargeError: If the payload or decoded image exceeds the configured limit.
        ImageDecodeError: If the bytes cannot be decoded into an image.
    """
    if len(image) > max_bytes:
        raise ImageTooLargeError(f"Image payload exceeds {max_bytes} bytes")
    buffer = np.frombuffer(image, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ImageDecodeError("Cannot decode image bytes")
    height, width = decoded.shape[:2]
    if height * width > max_pixels:
        raise ImageTooLargeError(f"Image size {width}x{height} exceeds {max_pixels} pixels")
    return decoded
