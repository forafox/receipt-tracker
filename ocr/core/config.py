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
        tesseract_config: Extra Tesseract configuration flags.
        quality_check_enabled: Whether to reject blurry, flat or glare-affected images.
        quality_min_sharpness: Minimum variance of the Laplacian of the grayscale image.
        quality_min_contrast: Minimum standard deviation of grayscale pixel values.
        quality_max_glare_ratio: Maximum share of near-saturated pixels.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="OCR_", extra="ignore")

    engine: str = "tesseract"
    tesseract_languages: str = "rus+eng"
    easyocr_languages: list[str] = Field(default_factory=lambda: ["ru", "en"])
    min_confidence: float = Field(default=0.0, ge=0, le=1)
    gpu: bool = False
    tesseract_config: str = ""
    quality_check_enabled: bool = True
    quality_min_sharpness: float = Field(default=50.0, ge=0)
    quality_min_contrast: float = Field(default=20.0, ge=0)
    quality_max_glare_ratio: float = Field(default=0.35, ge=0, le=1)
