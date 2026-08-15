"""Markup and web-format converters (HTML, Markdown, plain text, URLs).

Format names here use the registry's canonical spelling (``md``, ``txt``) --
declaring ``markdown``/``text`` instead would make these pairs unreachable from
any lookup that goes through :func:`~universal_converter.registry.detect_format`,
which normalises the aliases.
"""

import html as html_module
import re
from pathlib import Path
from typing import Any, Dict, Optional

from . import BaseConverter, ConversionResult, ConversionTask


class NetworkConverter(BaseConverter):
    """Converter for HTML / Markdown / plain text and URL introspection."""

    SUPPORTED_CONVERSIONS = {
        "html": ["md", "txt"],
        "md": ["html", "txt"],
        "txt": ["html", "md"],
        "url": ["json"],
    }
    PRIORITY = 30

    def convert(self, task: ConversionTask) -> ConversionResult:
        source = task.source_format.lower()
        target = task.target_format.lower()

        try:
            if source == "html" and target in ("md", "txt"):
                return self._from_html(task)
            if source == "md" and target in ("html", "txt"):
                return self._from_markdown(task)
            if source == "txt" and target in ("html", "md"):
                return self._from_text(task)
            if source == "url" and target == "json":
                return self._url_convert(task)

            return ConversionResult(
                success=False, error=f"Unsupported: {source} -> {target}"
            )
        except Exception as exc:
            return ConversionResult(success=False, error=str(exc))

    # --- html ------------------------------------------------------------

    def _from_html(self, task: ConversionTask) -> ConversionResult:
        content = Path(task.source_path).read_text(encoding="utf-8")
        if task.target_format.lower() == "md":
            out = self.html_to_markdown(content)
        else:
            out = self.html_to_text(content)
        Path(task.target_path).write_text(out + "\n", encoding="utf-8")
        return ConversionResult(success=True, output_path=str(task.target_path))

    @staticmethod
    def _strip_noise(html: str) -> str:
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.I)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.I)
        return re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    @classmethod
    def html_to_text(cls, html: str) -> str:
        """Strip tags and collapse blank runs, leaving readable plain text."""
        text = cls._strip_noise(html)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = re.sub(r"</(p|div|li|h[1-6]|tr)>", "\n", text, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = html_module.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    @staticmethod
    def _table_to_markdown(match: "re.Match") -> str:
        """Turn one ``<table>`` into a GitHub-flavoured Markdown table."""
        rows = []
        for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", match.group(1), re.DOTALL | re.I):
            cells = re.findall(
                r"<t([hd])[^>]*>(.*?)</t\1>", row_html, re.DOTALL | re.I
            )
            if cells:
                rows.append(
                    [
                        (
                            kind.lower(),
                            html_module.unescape(
                                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))
                            ).strip().replace("|", r"\|"),
                        )
                        for kind, text in cells
                    ]
                )
        if not rows:
            return ""

        width = max(len(row) for row in rows)
        # A leading all-<th> row is the header; otherwise synthesise one, since
        # Markdown tables cannot start straight at the body.
        if all(kind == "h" for kind, _ in rows[0]) and len(rows) > 1:
            header = [text for _, text in rows[0]]
            body = rows[1:]
        else:
            header = [f"col{i + 1}" for i in range(width)]
            body = rows
        header += [""] * (width - len(header))

        lines = [
            "| " + " | ".join(header) + " |",
            "|" + "|".join([" --- "] * width) + "|",
        ]
        for row in body:
            cells = [text for _, text in row] + [""] * (width - len(row))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n\n" + "\n".join(lines) + "\n\n"

    @classmethod
    def html_to_markdown(cls, html: str) -> str:
        """Convert the common inline/block HTML subset to Markdown."""
        md = cls._strip_noise(html)
        md = re.sub(r"<head[^>]*>.*?</head>", "", md, flags=re.DOTALL | re.I)
        md = re.sub(
            r"<table[^>]*>(.*?)</table>", cls._table_to_markdown, md, flags=re.DOTALL | re.I
        )
        for level in range(1, 7):
            md = re.sub(
                rf"<h{level}[^>]*>(.*?)</h{level}>",
                lambda m, lv=level: f"\n{'#' * lv} {m.group(1).strip()}\n",
                md,
                flags=re.DOTALL | re.I,
            )
        md = re.sub(r"<(strong|b)[^>]*>(.*?)</\1>", r"**\2**", md, flags=re.DOTALL | re.I)
        md = re.sub(r"<(em|i)[^>]*>(.*?)</\1>", r"*\2*", md, flags=re.DOTALL | re.I)
        md = re.sub(r"<pre[^>]*>(.*?)</pre>", r"\n```\n\1\n```\n", md, flags=re.DOTALL | re.I)
        md = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", md, flags=re.DOTALL | re.I)
        md = re.sub(
            r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r"[\2](\1)", md, flags=re.DOTALL | re.I
        )
        md = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", md, flags=re.DOTALL | re.I)
        md = re.sub(r"<br\s*/?>", "  \n", md, flags=re.I)
        md = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\1\n", md, flags=re.DOTALL | re.I)
        md = re.sub(r"<[^>]+>", "", md)
        md = html_module.unescape(md)
        md = re.sub(r"[ \t]+\n", "\n", md)
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()

    # --- markdown --------------------------------------------------------

    def _from_markdown(self, task: ConversionTask) -> ConversionResult:
        content = Path(task.source_path).read_text(encoding="utf-8")
        if task.target_format.lower() == "html":
            out = self.markdown_to_html(content)
        else:
            out = self.markdown_to_text(content)
        Path(task.target_path).write_text(out + "\n", encoding="utf-8")
        return ConversionResult(success=True, output_path=str(task.target_path))

    @staticmethod
    def markdown_to_html(markdown: str) -> str:
        """Render the common Markdown subset (headings, emphasis, code,
        links, lists, paragraphs) as a standalone HTML document."""
        out_lines = []
        in_list = False
        in_code = False

        def inline(text: str) -> str:
            text = html_module.escape(text, quote=False)
            text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
            text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
            text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
            text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
            return text

        for raw in markdown.splitlines():
            if raw.strip().startswith("```"):
                if in_list:
                    out_lines.append("</ul>")
                    in_list = False
                out_lines.append("</pre>" if in_code else "<pre>")
                in_code = not in_code
                continue
            if in_code:
                out_lines.append(html_module.escape(raw))
                continue

            line = raw.rstrip()
            heading = re.match(r"^(#{1,6})\s+(.*)$", line)
            bullet = re.match(r"^\s*[-*+]\s+(.*)$", line)

            if bullet:
                if not in_list:
                    out_lines.append("<ul>")
                    in_list = True
                out_lines.append(f"<li>{inline(bullet.group(1))}</li>")
                continue
            if in_list:
                out_lines.append("</ul>")
                in_list = False

            if heading:
                level = len(heading.group(1))
                out_lines.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            elif not line.strip():
                continue
            else:
                out_lines.append(f"<p>{inline(line)}</p>")

        if in_list:
            out_lines.append("</ul>")
        if in_code:
            out_lines.append("</pre>")

        body = "\n".join(out_lines)
        return (
            '<!DOCTYPE html>\n<html lang="en">\n<head><meta charset="utf-8">'
            "<title>Converted</title></head>\n<body>\n"
            f"{body}\n</body>\n</html>"
        )

    @classmethod
    def markdown_to_text(cls, markdown: str) -> str:
        """Flatten Markdown to plain prose by rendering then stripping tags."""
        return cls.html_to_text(cls.markdown_to_html(markdown))

    # --- plain text ------------------------------------------------------

    def _from_text(self, task: ConversionTask) -> ConversionResult:
        content = Path(task.source_path).read_text(encoding="utf-8")
        if task.target_format.lower() == "html":
            out = self.text_to_html(content)
        else:
            # Escape the characters that would otherwise be read as markup.
            out = re.sub(r"([\\`*_\[\]#])", r"\\\1", content)
        Path(task.target_path).write_text(out + "\n", encoding="utf-8")
        return ConversionResult(success=True, output_path=str(task.target_path))

    @staticmethod
    def text_to_html(text: str) -> str:
        """Wrap plain text in a minimal HTML document, one <p> per block."""
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        body = "\n".join(
            "<p>" + html_module.escape(b).replace("\n", "<br>\n") + "</p>"
            for b in blocks
        )
        return (
            '<!DOCTYPE html>\n<html lang="en">\n<head><meta charset="utf-8">'
            "<title>Converted</title></head>\n<body>\n"
            f"{body}\n</body>\n</html>"
        )

    # --- urls ------------------------------------------------------------

    def _url_convert(self, task: ConversionTask) -> ConversionResult:
        import json
        from urllib.parse import parse_qs, urlparse

        content = Path(task.source_path).read_text(encoding="utf-8").strip()
        parsed = urlparse(content)

        url_data: Dict[str, Any] = {
            "url": content,
            "scheme": parsed.scheme,
            "netloc": parsed.netloc,
            "hostname": parsed.hostname,
            "port": parsed.port,
            "path": parsed.path,
            "params": parsed.params,
            "query": parse_qs(parsed.query),
            "fragment": parsed.fragment,
        }

        Path(task.target_path).write_text(
            json.dumps(url_data, indent=2), encoding="utf-8"
        )
        return ConversionResult(success=True, output_path=str(task.target_path))

    def convert_format(
        self, path: str, to_format: str, output: Optional[str] = None
    ) -> ConversionResult:
        """Convert a markup file to ``to_format``."""
        task = ConversionTask(
            source_path=Path(path),
            target_path=Path(output or Path(path).with_suffix(f".{to_format}")),
            source_format=Path(path).suffix[1:],
            target_format=to_format,
        )
        return self.convert(task)
