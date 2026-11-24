from pydantic import Field, field_validator
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
        adaptive: Whether Tesseract should run several page-segmentation modes and let the
            pipeline choose the best parsed result.
        adaptive_psm_modes: Page-segmentation modes tried when ``adaptive`` is enabled.
        quality_check_enabled: Whether to reject blurry, flat or glare-affected images.
        quality_min_sharpness: Minimum variance of the Laplacian of the grayscale image.
        quality_min_contrast: Minimum standard deviation of grayscale pixel values.
        quality_max_glare_ratio: Maximum share of near-saturated pixels.
        postprocess_enabled: Whether to filter tiny and overlapping blocks and sort them.
        postprocess_min_box_area: Minimum axis-aligned box area in pixels to keep a block.
        postprocess_nms_iou: Maximum IoU with a stronger block before a block is suppressed.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="OCR_", extra="ignore")

    engine: str = "tesseract"
    tesseract_languages: str = "rus+eng"
    easyocr_languages: list[str] = Field(default_factory=lambda: ["ru", "en"])
    min_confidence: float = Field(default=0.0, ge=0, le=1)
    gpu: bool = False
    tesseract_config: str = ""
    adaptive: bool = False
    adaptive_psm_modes: list[int] = Field(default_factory=lambda: [3, 4, 6])
    quality_check_enabled: bool = True
    quality_min_sharpness: float = Field(default=50.0, ge=0)
    quality_min_contrast: float = Field(default=20.0, ge=0)
    quality_max_glare_ratio: float = Field(default=0.35, ge=0, le=1)
    postprocess_enabled: bool = True
    postprocess_min_box_area: float = Field(default=9.0, ge=0)
    postprocess_nms_iou: float = Field(default=0.5, gt=0, le=1)

    @field_validator("adaptive_psm_modes")
    @classmethod
    def _validate_psm_modes(cls, modes: list[int]) -> list[int]:
        if not modes:
            msg = "adaptive_psm_modes must not be empty"
            raise ValueError(msg)
        if any(not 1 <= mode <= 13 for mode in modes):
            msg = "adaptive_psm_modes entries must be between 1 and 13"
            raise ValueError(msg)
        return list(dict.fromkeys(modes))
