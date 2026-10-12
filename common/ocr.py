from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

Point = tuple[float, float]


class TextBlock(BaseModel):
    """A recognized text fragment with its position on a receipt image.

    Attributes:
        text: Recognized text of the fragment.
        bbox: Quadrilateral around the text as absolute-pixel ``(x, y)`` points, ordered
            clockwise starting from the top-left corner.
        confidence: Recognition confidence in the ``0..1`` range, or None when the engine
            does not report one.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    bbox: list[Point] = Field(min_length=4, max_length=4)
    confidence: float | None = Field(default=None, ge=0, le=1)


class OcrResult(BaseModel):
    """OCR output for a single receipt image.

    Attributes:
        blocks: Recognized text fragments in engine order.
        image_width: Source image width in pixels, if known.
        image_height: Source image height in pixels, if known.
    """

    model_config = ConfigDict(extra="forbid")

    blocks: list[TextBlock] = Field(default_factory=list)
    image_width: int | None = Field(default=None, gt=0)
    image_height: int | None = Field(default=None, gt=0)


class OcrEngine(Protocol):
    """Interface for engines that locate and read text on a receipt image.

    Engines receive raw image bytes and return text fragments together with their
    bounding boxes. ``recognize`` matches ``pipeline.TextRecognizer`` so an engine can be
    passed to the receipt pipeline without an adapter.
    """

    name: str

    def recognize_blocks(self, image: bytes) -> OcrResult:
        """Recognize receipt text with bounding boxes.

        Args:
            image: Raw receipt image bytes.

        Returns:
            Recognized text fragments with their absolute-pixel bounding boxes.
        """
        ...

    def recognize(self, image: bytes) -> str:
        """Recognize receipt text as a plain string.

        Args:
            image: Raw receipt image bytes.

        Returns:
            Recognized text with one receipt line per text line.
        """
        ...
