import pytest

from app.core.exception import UnprocessableError
from app.services.text_extraction import PDF, extract, media_type_for, normalize
from tests.pdf import make_pdf


def test_pdf_pages_and_offsets() -> None:
    extracted = extract(make_pdf(["First page text.", "Second page text."]), PDF)
    assert extracted.page_starts is not None
    first, second = extracted.page_starts
    assert extracted.content[first:].startswith("First page text.")
    assert extracted.content[second:].startswith("Second page text.")


def test_plain_text_has_no_pages() -> None:
    extracted = extract(b"\xef\xbb\xbfHello\r\n\r\n\r\n\r\nworld\t\tagain", "text/plain")
    assert extracted.content == "Hello\n\nworld again"
    assert extracted.page_starts is None


@pytest.mark.parametrize(
    ("raw", "media_type", "message"),
    [
        (b"not a pdf", PDF, "The PDF could not be read"),
        (b"\xff\xfe\x00bad", "text/plain", "Text files must be UTF-8"),
        (b"   \n  ", "text/plain", "No extractable text found"),
    ],
)
def test_unreadable_input_is_rejected(raw: bytes, media_type: str, message: str) -> None:
    with pytest.raises(UnprocessableError, match=message):
        extract(raw, media_type)


def test_media_type_prefers_extension() -> None:
    assert media_type_for("notes.md", "application/octet-stream") == "text/markdown"
    assert media_type_for("report.PDF", None) == PDF
    assert media_type_for("readme", "text/plain; charset=utf-8") == "text/plain"
    with pytest.raises(UnprocessableError):
        media_type_for("sheet.xlsx", "application/vnd.ms-excel")


def test_normalize_collapses_whitespace() -> None:
    assert normalize("a  \t b\x00\n\n\n\nc ") == "a b\n\nc"
