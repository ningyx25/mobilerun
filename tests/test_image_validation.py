"""Tests for the structural image-completeness check.

A truncated screenshot (a cut ``adb exec-out`` stream) is a strict prefix of a
valid file: PNG dimensions still parse from the IHDR header, and only a pixel
decode fails. ``is_complete_image`` catches it up front by looking for the
end-of-image marker near the tail.
"""

import io

import pytest
from PIL import Image

from mobilerun.tools.helpers.images import (
    IncompleteImageError,
    ensure_complete_image,
    is_complete_image,
)


def _png(width=20, height=20):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg(width=20, height=20):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(1, 2, 3)).save(buf, format="JPEG")
    return buf.getvalue()


def test_valid_png_is_complete():
    assert is_complete_image(_png())


def test_truncated_png_is_incomplete():
    data = _png()
    assert not is_complete_image(data[: len(data) // 2])
    # Even cutting a single byte breaks the trailing IEND chunk.
    assert not is_complete_image(data[: len(data) - 1])


def test_png_with_trailing_bytes_is_still_complete():
    # The tail window tolerates junk appended after IEND: a strict suffix test
    # would reject every frame if a writer ever appended one byte.
    data = _png()
    assert is_complete_image(data + b"\n")
    assert is_complete_image(data + b"junk")


def test_empty_header_only_and_garbage_are_incomplete():
    assert not is_complete_image(b"")
    assert not is_complete_image(_png()[:8])  # magic only
    assert not is_complete_image(b"not an image at all")


def test_valid_jpeg_is_complete():
    assert is_complete_image(_jpeg())


def test_truncated_jpeg_is_incomplete():
    data = _jpeg()
    assert not is_complete_image(data[: len(data) // 2])
    assert not is_complete_image(data[: len(data) - 1])


def test_ensure_complete_image_reports_size_and_passes_valid_bytes():
    data = _png()
    assert ensure_complete_image(data, source="/screenshot") is data

    with pytest.raises(IncompleteImageError, match=r"\(12 bytes\)"):
        ensure_complete_image(b"x" * 12, source="/screenshot b64_png")
