from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OcrSettings(BaseSettings):
    """OCR engine selection loaded from environment variables.

    Attributes:
        engine: Concrete engine to build, either ``tesseract`` or ``easyocr``.
        tesseract_languages: Tesseract language string, for example ``rus+eng``.
        easyocr_languages: EasyOCR language codes passed to the reader.
        min_confidence: Minimum block confidence in the ``0..1`` range to keep a block.
        gpu: Whether EasyOCR should use the GPU.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="OCR_")

    engine: str = "tesseract"
    tesseract_languages: str = "rus+eng"
    easyocr_languages: list[str] = Field(default_factory=lambda: ["ru", "en"])
    min_confidence: float = Field(default=0.0, ge=0, le=1)
    gpu: bool = False
