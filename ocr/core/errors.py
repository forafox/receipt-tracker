class OcrError(Exception):
    """Base error for OCR failures."""


class ImageDecodeError(OcrError):
    """Raised when input bytes cannot be decoded into an image."""


class ImageTooLargeError(ImageDecodeError):
    """Raised when an input image exceeds the allowed payload or pixel size."""


class ImageQualityError(ImageDecodeError):
    """Raised when an image is too blurry, flat or glare-affected for recognition.

    Quality failures inherit from ``ImageDecodeError`` because the pipeline reports
    them as an unreadable receipt that should be re-shot, not as an engine outage.
    """


class OcrEngineUnavailableError(OcrError):
    """Raised when an OCR engine or its runtime dependency is unavailable."""
