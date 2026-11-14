from itertools import permutations

import pytest

from common.ocr import OcrResult, TextBlock
from extract.layout import reconstruct_layout


def _block(
    text: str,
    left: float,
    top: float,
    right: float,
    bottom: float,
    confidence: float | None = None,
) -> TextBlock:
    return TextBlock(
        text=text,
        bbox=[(left, top), (right, top), (right, bottom), (left, bottom)],
        confidence=confidence,
    )


def test_empty_result_has_no_layout() -> None:
    assert reconstruct_layout(OcrResult()) == []


def test_single_entire_row_is_retained_intact_without_image_dimensions() -> None:
    block = _block("  ХЛЕБ ДЕРЕВЕНСКИЙ   50.00  ", 10, 20, 270, 34)
    result = OcrResult(blocks=[block])
    original = result.model_dump()

    lines = reconstruct_layout(result)

    assert len(lines) == 1
    line = lines[0]
    assert line.blocks == (block,)
    assert line.blocks[0] is block
    assert line.blocks[0].confidence is None
    assert line.text == block.text
    assert (line.x_min, line.y_min, line.x_max, line.y_max) == (10, 20, 270, 34)
    assert result.model_dump() == original


@pytest.mark.parametrize("scale", [0.25, 4.0])
def test_shuffled_receipt_retains_reading_order_and_column_positions(scale: float) -> None:
    name_first = _block("КАПУЧИНО", 10, 60, 110, 72, 0.96)
    amount_first = _block("250.00", 210, 61, 270, 73, 0.99)
    name_second = _block("КРУАССАН", 10, 85, 110, 97, 0.95)
    amount_second = _block("150.00", 210, 84, 270, 96, 0.98)
    blocks = [amount_second, name_first, amount_first, name_second]
    for block in blocks:
        block.bbox = [(x * scale, y * scale) for x, y in block.bbox]
    result = OcrResult(blocks=blocks, image_width=int(300 * scale), image_height=int(120 * scale))
    original = result.model_dump()

    lines = reconstruct_layout(result)

    assert [line.text for line in lines] == ["КАПУЧИНО 250.00", "КРУАССАН 150.00"]
    assert all(
        actual is expected
        for line, expected_blocks in zip(
            lines, [(name_first, amount_first), (name_second, amount_second)], strict=True
        )
        for actual, expected in zip(line.blocks, expected_blocks, strict=True)
    )
    assert [block.confidence for line in lines for block in line.blocks] == [
        0.96,
        0.99,
        0.95,
        0.98,
    ]
    assert (lines[0].x_min, lines[0].y_min, lines[0].x_max, lines[0].y_max) == (
        10 * scale,
        60 * scale,
        270 * scale,
        73 * scale,
    )
    assert max(x for x, _ in name_first.bbox) < min(x for x, _ in amount_first.bbox)
    assert min(x for x, _ in amount_first.bbox) == min(x for x, _ in amount_second.bbox)
    assert lines[0].y_max < lines[1].y_min
    for order in permutations(blocks):
        assert reconstruct_layout(OcrResult(blocks=list(order))) == lines
    assert reconstruct_layout(result) == lines
    assert result.model_dump() == original


def test_multiline_product_preserves_physical_rows_and_left_to_right_fragments() -> None:
    blocks = [
        _block("179.90", 210, 42, 270, 54),
        _block("45% 200Г", 10, 22, 110, 34),
        _block("РОССИЙСКИЙ", 60, 2, 160, 14),
        _block("1 x 179.90", 10, 42, 100, 54),
        _block("СЫР", 10, 0, 50, 12),
    ]

    lines = reconstruct_layout(OcrResult(blocks=blocks))

    assert [line.text for line in lines] == [
        "СЫР РОССИЙСКИЙ",
        "45% 200Г",
        "1 x 179.90 179.90",
    ]
    assert [len(line.blocks) for line in lines] == [2, 1, 2]
    assert [block.text for block in lines[0].blocks] == ["СЫР", "РОССИЙСКИЙ"]


def test_vertical_overlap_does_not_chain_into_neighboring_rows() -> None:
    blocks = [
        _block("Первая", 10, 0, 80, 12),
        _block("смещённая", 100, 4, 180, 16),
        _block("Вторая", 10, 8, 80, 20),
        _block("строка", 100, 9, 180, 21),
    ]

    lines = reconstruct_layout(OcrResult(blocks=blocks[::-1]))

    assert [line.text for line in lines] == ["Первая смещённая", "Вторая строка"]


def test_tall_block_does_not_absorb_short_neighboring_rows() -> None:
    blocks = [
        _block("Высокий фрагмент", 10, 0, 100, 40),
        _block("Первая", 150, 2, 200, 12),
        _block("Вторая", 150, 24, 200, 34),
    ]

    assert [line.text for line in reconstruct_layout(OcrResult(blocks=blocks))] == [
        "Высокий фрагмент",
        "Первая",
        "Вторая",
    ]


def test_tilted_quadrilateral_uses_all_four_corners() -> None:
    block = TextBlock(text="Наклонный текст", bbox=[(12, 20), (90, 16), (94, 32), (8, 35)])

    line = reconstruct_layout(OcrResult(blocks=[block]))[0]

    assert (line.x_min, line.y_min, line.x_max, line.y_max) == (8, 16, 94, 35)
    assert line.blocks[0] is block
    assert line.blocks[0].bbox == [(12, 20), (90, 16), (94, 32), (8, 35)]


def test_degenerate_boxes_retain_geometry_without_division_by_zero() -> None:
    blocks = [
        _block("Точка", 10, 0, 10, 0),
        _block("Плоский", 30, 0, 60, 0),
        _block("Слева", 10, 10, 50, 22),
        _block("Нулевая ширина", 100, 10, 100, 22),
    ]

    lines = reconstruct_layout(OcrResult(blocks=blocks[::-1]))

    assert [line.text for line in lines] == ["Точка", "Плоский", "Слева Нулевая ширина"]
    assert (lines[0].x_min, lines[0].y_min, lines[0].x_max, lines[0].y_max) == (10, 0, 10, 0)
    assert (lines[1].x_min, lines[1].y_min, lines[1].x_max, lines[1].y_max) == (30, 0, 60, 0)
    assert len(lines[2].blocks) == 2


def test_equal_geometry_uses_text_and_confidence_as_deterministic_ties() -> None:
    blocks = [
        _block("Б", 10, 20, 50, 32, 0.9),
        _block("А", 10, 20, 50, 32, 0.9),
        _block("А", 10, 20, 50, 32),
    ]
    expected = reconstruct_layout(OcrResult(blocks=blocks))

    assert [block.text for block in expected[0].blocks] == ["А", "А", "Б"]
    assert [block.confidence for block in expected[0].blocks] == [None, 0.9, 0.9]
    for order in permutations(blocks):
        assert reconstruct_layout(OcrResult(blocks=list(order))) == expected


def test_nonfinite_coordinates_are_rejected_without_inventing_geometry() -> None:
    block = _block("Неизвестная позиция", 0, 0, float("nan"), 12)

    with pytest.raises(ValueError, match="coordinates must be finite"):
        reconstruct_layout(OcrResult(blocks=[block]))
