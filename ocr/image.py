from __future__ import annotations

import cv2
import numpy as np

from ocr.errors import ImageDecodeError


def decode_image(image: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR ``numpy`` array.

    Args:
        image: Encoded image bytes (JPEG, PNG, ...).

    Returns:
        Decoded BGR image array.

    Raises:
        ImageDecodeError: If the bytes cannot be decoded into an image.
    """
    buffer = np.frombuffer(image, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ImageDecodeError("Cannot decode image bytes")
    return decoded
