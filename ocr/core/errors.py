class OcrError(Exception):
    """Base error for OCR failures."""


class ImageDecodeError(OcrError):
    """Raised when input bytes cannot be decoded into an image."""


class ImageTooLargeError(ImageDecodeError):
    """Raised when an input image exceeds the allowed payload or pixel size."""


class OcrEngineUnavailableError(OcrError):
    """Raised when an OCR engine or its runtime dependency is unavailable."""
