"""Document conversion module (PDF, DOCX, and text-shaped inputs).

Fidelity is deliberately text-level: paragraphs and heading levels survive,
styling and layout do not. Anything that needs true layout fidelity wants
LibreOffice or pandoc, and this module says so rather than pretending.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from . import BaseConverter, ConversionResult, ConversionTask

_DOCX_HINT = "python-docx required. Install: pip install universal-converter[docx]"
_PDF_READ_HINT = "pypdf required. Install: pip install universal-converter[pdf]"
_PDF_WRITE_HINT = "reportlab required. Install: pip install universal-converter[pdf]"

#: (source, target) -> importable modules the pair needs.
_PAIR_REQUIREMENTS = {
    ("pdf", "txt"): ["pypdf"],
    ("pdf", "docx"): ["pypdf", "docx"],
    ("docx", "txt"): ["docx"],
    ("docx", "md"): ["docx"],
    ("docx", "html"): ["docx"],
    ("docx", "pdf"): ["docx", "reportlab"],
    ("txt", "docx"): ["docx"],
    ("md", "docx"): ["docx"],
    ("html", "docx"): ["docx"],
    ("txt", "pdf"): ["reportlab"],
    ("md", "pdf"): ["reportlab"],
    ("html", "pdf"): ["reportlab"],
}


class DocumentConverter(BaseConverter):
    """Converter for PDF/DOCX documents and text-to-document rendering."""

    SUPPORTED_CONVERSIONS = {
        "pdf": ["txt", "docx"],
        "docx": ["txt", "md", "html", "pdf"],
        "txt": ["docx", "pdf"],
        "md": ["docx", "pdf"],
        "html": ["docx", "pdf"],
    }
    PRIORITY = 20
    REQUIRES_PYTHON = ["docx", "pypdf", "reportlab"]

    @classmethod
    def requirements_for(cls, source_format: str, target_format: str) -> List[str]:
        pair = (source_format.lower(), target_format.lower())
        return list(_PAIR_REQUIREMENTS.get(pair, cls.REQUIRES_PYTHON))

    def convert(self, task: ConversionTask) -> ConversionResult:
        source = task.source_format.lower()
        target = task.target_format.lower()

        try:
            if source == "pdf" and target in ("txt", "docx"):
                return self._from_pdf(task)
            if source == "docx":
                return self._from_docx(task)
            if source in ("txt", "md", "html") and target in ("docx", "pdf"):
                return self._from_text_like(task)

            return ConversionResult(
                success=False, error=f"Unsupported: {source} -> {target}"
            )
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc))

    # --- pdf -------------------------------------------------------------

    def _from_pdf(self, task: ConversionTask) -> ConversionResult:
        try:
            import pypdf
        except ImportError:
            return ConversionResult(success=False, error=_PDF_READ_HINT)

        reader = pypdf.PdfReader(str(task.source_path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)

        if task.target_format.lower() == "txt":
            Path(task.target_path).write_text(text, encoding="utf-8")
            return ConversionResult(
                success=True,
                output_path=str(task.target_path),
                metadata={"pages": len(reader.pages)},
            )
        return self._write_docx(
            [(0, line) for line in text.split("\n")], task.target_path
        )

    # --- docx ------------------------------------------------------------

    def _from_docx(self, task: ConversionTask) -> ConversionResult:
        try:
            from docx import Document
        except ImportError:
            return ConversionResult(success=False, error=_DOCX_HINT)

        doc = Document(str(task.source_path))
        blocks = [
            (self._heading_level(paragraph.style.name), paragraph.text)
            for paragraph in doc.paragraphs
        ]
        target = task.target_format.lower()

        if target == "txt":
            body = "\n".join(text for _, text in blocks)
        elif target == "md":
            body = "\n\n".join(
                (f"{'#' * level} {text}" if level else text)
                for level, text in blocks
                if text.strip()
            )
        elif target == "html":
            from html import escape

            rendered = "\n".join(
                (
                    f"<h{level}>{escape(text)}</h{level}>"
                    if level
                    else f"<p>{escape(text)}</p>"
                )
                for level, text in blocks
                if text.strip()
            )
            body = (
                '<!DOCTYPE html>\n<html lang="en">\n<head><meta charset="utf-8">'
                "<title>Converted</title></head>\n<body>\n"
                f"{rendered}\n</body>\n</html>"
            )
        else:  # pdf
            return self._write_pdf(blocks, task.target_path)

        Path(task.target_path).write_text(body + "\n", encoding="utf-8")
        return ConversionResult(
            success=True,
            output_path=str(task.target_path),
            metadata={"paragraphs": len(blocks)},
        )

    @staticmethod
    def _heading_level(style_name: Optional[str]) -> int:
        """Map a python-docx style name to a heading level (0 = body text)."""
        match = re.match(r"Heading (\d)", style_name or "")
        return int(match.group(1)) if match else 0

    # --- text-shaped sources --------------------------------------------

    def _from_text_like(self, task: ConversionTask) -> ConversionResult:
        content = Path(task.source_path).read_text(encoding="utf-8")
        source = task.source_format.lower()

        if source == "html":
            from .network import NetworkConverter

            content = NetworkConverter.html_to_markdown(content)
        blocks = self._markdown_blocks(content) if source in ("html", "md") else [
            (0, line) for line in content.splitlines()
        ]

        if task.target_format.lower() == "docx":
            return self._write_docx(blocks, task.target_path)
        return self._write_pdf(blocks, task.target_path)

    @staticmethod
    def _markdown_blocks(markdown: str) -> List[Tuple[int, str]]:
        """Split Markdown into ``(heading_level, text)`` blocks."""
        blocks: List[Tuple[int, str]] = []
        for line in markdown.splitlines():
            heading = re.match(r"^(#{1,6})\s+(.*)$", line)
            if heading:
                blocks.append((len(heading.group(1)), heading.group(2).strip()))
            else:
                blocks.append((0, line))
        return blocks

    # --- writers ---------------------------------------------------------

    def _write_docx(
        self, blocks: List[Tuple[int, str]], path: Path
    ) -> ConversionResult:
        try:
            from docx import Document
        except ImportError:
            return ConversionResult(success=False, error=_DOCX_HINT)

        doc = Document()
        for level, text in blocks:
            if level:
                doc.add_heading(text, level=min(level, 9))
            else:
                doc.add_paragraph(text)
        doc.save(str(path))
        return ConversionResult(
            success=True, output_path=str(path), metadata={"paragraphs": len(blocks)}
        )

    def _write_pdf(self, blocks: List[Tuple[int, str]], path: Path) -> ConversionResult:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.utils import simpleSplit
            from reportlab.pdfgen import canvas
        except ImportError:
            return ConversionResult(success=False, error=_PDF_WRITE_HINT)

        pdf = canvas.Canvas(str(path), pagesize=letter)
        width, height = letter
        margin, y = 50, letter[1] - 50
        pages = 1

        for level, text in blocks:
            font, size = ("Helvetica-Bold", max(18 - 2 * level, 11)) if level else (
                "Helvetica",
                11,
            )
            pdf.setFont(font, size)
            for line in simpleSplit(text, font, size, width - 2 * margin) or [""]:
                if y < margin:
                    pdf.showPage()
                    pages += 1
                    pdf.setFont(font, size)
                    y = height - margin
                pdf.drawString(margin, y, line)
                y -= size + 4
            y -= 4

        pdf.save()
        return ConversionResult(
            success=True, output_path=str(path), metadata={"pages": pages}
        )

    def convert_format(
        self, path: str, to_format: str, output: Optional[str] = None
    ) -> ConversionResult:
        """Convert a document to ``to_format``."""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format,
        )
        return self.convert(task)
