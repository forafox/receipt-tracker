import pytest

# These expectations describe how the M2 parser behaved before commit 9d16295
# ("Add Hunyuan OCR receipt parsing"), which widened the parsing rules for Hunyuan
# output. The M2 and M3 owners have to decide which behaviour is correct. strict=True
# makes the test fail as soon as the old behaviour is back, so the mark gets removed.
HUNYUAN_PARSER_CHANGE = pytest.mark.xfail(
    strict=True,
    reason="M2 parser behaviour changed in 9d16295 (Hunyuan OCR); pending M2/M3 decision",
)
