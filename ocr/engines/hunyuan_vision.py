from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request

from common.ocr import OcrResult, TextBlock
from ocr.core.errors import OcrEngineUnavailableError
from ocr.core.image import decode_image

_PROMPT = (
    "提取图中的文字。 Это русский кассовый чек. Перепиши все читаемые физические строки "
    "сверху вниз, сохраняя русский текст, числа, цены и порядок строк. Не пропускай товары, "
    "итоги, реквизиты и нижнюю часть чека. Не переводи, не исправляй и не додумывай текст. "
    "Каждую строку выведи отдельно, без пояснений."
)


class HunyuanVisionEngine:
    """Receipt transcription with HunyuanOCR through a local llama.cpp server."""

    name = "hunyuan_ocr"

    def __init__(
        self,
        *,
        model: str = "HunyuanOCR",
        url: str = "http://localhost:8231/v1/chat/completions",
        timeout_seconds: float = 600.0,
    ) -> None:
        """Create an HunyuanOCR client.

        Args:
            model: Model alias exposed by llama.cpp.
            url: OpenAI-compatible llama.cpp chat-completions endpoint.
            timeout_seconds: Maximum time to wait for one receipt transcription.
        """
        self._model = model
        self._url = url
        self._timeout_seconds = timeout_seconds

    def recognize_blocks(self, image: bytes) -> OcrResult:
        """Transcribe the image and adapt ordered rows to the M1/M2 OCR contract.

        Hunyuan's plain-text mode reliably returns complete receipt rows, while its
        generated spot-coordinate JSON was unstable in the evaluation receipt. The
        adapter therefore preserves reading order and assigns coarse synthetic row boxes;
        M2 still receives the standard ``OcrResult`` and can parse complete item rows.

        Args:
            image: Encoded receipt photo.

        Returns:
            One text block per nonempty transcript row with synthetic top-to-bottom boxes.

        Raises:
            ImageDecodeError: If the image is invalid or exceeds configured limits.
            OcrEngineUnavailableError: If the local server fails or returns no text.
        """
        decoded = decode_image(image)
        height, width = decoded.shape[:2]
        mime = _image_mime(image)
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
                },
            },
            {"type": "text", "text": _PROMPT},
        ]
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 4096,
            "stream": False,
        }
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                result = json.loads(response.read())
            text = result["choices"][0]["message"]["content"].strip()
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            raise OcrEngineUnavailableError(f"HunyuanOCR request failed: {error}") from error
        if not text:
            raise OcrEngineUnavailableError("HunyuanOCR returned an empty transcript")

        raw_lines = text.splitlines()
        rows = [(index, line.strip()) for index, line in enumerate(raw_lines) if line.strip()]
        if not rows:
            raise OcrEngineUnavailableError("HunyuanOCR returned no nonempty lines")
        row_height = max(1, height // max(2 * len(raw_lines), 1))
        blocks = []
        for raw_index, line in rows:
            y_min = min(height - 1, round(raw_index * height / max(len(raw_lines), 1)))
            y_max = min(height, max(y_min + 1, y_min + row_height))
            blocks.append(
                TextBlock(
                    text=line,
                    bbox=[(0, y_min), (width, y_min), (width, y_max), (0, y_max)],
                    confidence=None,
                )
            )
        return OcrResult(blocks=blocks, image_width=width, image_height=height)

    def recognize(self, image: bytes) -> str:
        """Return recognized rows in reading order.

        Args:
            image: Encoded receipt photo.

        Returns:
            One row per line.
        """
        return "\n".join(block.text for block in self.recognize_blocks(image).blocks)


def _image_mime(image: bytes) -> str:
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image.startswith(b"RIFF") and image[8:12] == b"WEBP":
        return "image/webp"
    if image.startswith(b"\xff\xd8"):
        return "image/jpeg"
    return mimetypes.guess_type("receipt.jpg")[0] or "image/jpeg"
