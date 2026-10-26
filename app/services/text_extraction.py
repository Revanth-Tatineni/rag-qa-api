"""Turn an uploaded file into plain text, keeping where each PDF page starts."""

import io
import logging
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exception import UnprocessableError

PDF = "application/pdf"
TEXT_TYPES = {"text/plain", "text/markdown", "text/x-markdown"}
_EXTENSION_TYPES = {".pdf": PDF, ".txt": "text/plain", ".md": "text/markdown"}

# pypdf logs a warning per font it can't fully decode; the text still extracts.
logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass(frozen=True)
class ExtractedText:
    content: str
    page_starts: list[int] | None


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def media_type_for(filename: str, declared: str | None) -> str:
    """Trust the extension over the browser's declared type, which is often octet-stream."""
    ext = filename[filename.rfind(".") :].lower() if "." in filename else ""
    media_type = _EXTENSION_TYPES.get(ext) or (declared or "").split(";")[0].strip().lower()
    if media_type != PDF and media_type not in TEXT_TYPES:
        raise UnprocessableError("Only PDF, Markdown and plain-text files are supported")
    return media_type


def _from_pdf(raw: bytes) -> ExtractedText:
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [normalize(page.extract_text() or "") for page in reader.pages]
    except (PdfReadError, ValueError, KeyError) as exc:
        raise UnprocessableError("The PDF could not be read") from exc
    parts: list[str] = []
    starts: list[int] = []
    offset = 0
    for page in pages:
        starts.append(offset)
        parts.append(page)
        offset += len(page) + 2  # the "\n\n" joining pages
    return ExtractedText("\n\n".join(parts), starts)


def extract(raw: bytes, media_type: str) -> ExtractedText:
    if media_type == PDF:
        extracted = _from_pdf(raw)
    else:
        try:
            extracted = ExtractedText(normalize(raw.decode("utf-8-sig")), None)
        except UnicodeDecodeError as exc:
            raise UnprocessableError("Text files must be UTF-8") from exc
    if not extracted.content.strip():
        # Usually a scanned PDF: images only, no text layer. OCR is out of scope.
        raise UnprocessableError("No extractable text found")
    return extracted
